from __future__ import absolute_import, division, print_function, unicode_literals

"""

Implementation of the qcore of pyanalyze.

Contains an AST visitor that visits each node and infers a value for most nodes.

Some things that are not yet checked and could be are listed in
https://app.asana.com/0/10206869882253/45193094024681.

"""

from six.moves import builtins, range, reduce
from abc import abstractmethod
import ast
from ast_decompiler import decompile

try:
    import asyncio
except ImportError:
    asyncio = None
import collections
import contextlib
import imp
import inspect
import inspect2
from itertools import chain
import logging
import os.path
import pickle
import random
import re
import six
import string
import sys
import tempfile
import traceback
import types
from typing import Iterable

import asynq
import qcore
from qcore.helpers import safe_str

from pyanalyze import analysis_lib
from pyanalyze import node_visitor
from pyanalyze.annotations import type_from_runtime, type_from_value, is_typing_name
from pyanalyze.arg_spec import (
    ArgSpecCache,
    BoundMethodArgSpecWrapper,
    is_dot_asynq_function,
)
from pyanalyze.config import Config
from pyanalyze.error_code import ErrorCode, DISABLED_BY_DEFAULT, ERROR_DESCRIPTION
from pyanalyze.find_unused import UnusedObjectFinder
from pyanalyze import format_strings
from pyanalyze import importer
from pyanalyze import method_return_type
from pyanalyze.stacked_scopes import (
    CompositeVariable,
    Constraint,
    AndConstraint,
    OrConstraint,
    NULL_CONSTRAINT,
    ConstraintType,
    ScopeType,
    StackedScopes,
    VisitorState,
    LEAVES_LOOP,
    LEAVES_SCOPE,
)
from pyanalyze.asynq_checker import AsyncFunctionKind, AsynqChecker
from pyanalyze.yield_checker import YieldChecker
from pyanalyze.value import (
    boolean_value,
    UNINITIALIZED_VALUE,
    UNRESOLVED_VALUE,
    NO_RETURN_VALUE,
    unite_values,
    KnownValue,
    TypedValue,
    MultiValuedValue,
    UnboundMethodValue,
    VariableNameValue,
    ReferencingValue,
    SubclassValue,
    DictIncompleteValue,
    SequenceIncompleteValue,
    AsyncTaskIncompleteValue,
    AwaitableIncompleteValue,
    GenericValue,
    TypedDictValue,
)

OPERATION_TO_DESCRIPTION_AND_METHOD = {
    ast.Add: ("addition", "__add__", "__radd__"),
    ast.Sub: ("subtraction", "__sub__", "__rsub__"),
    ast.Mult: ("multiplication", "__mul__", "__rmul__"),
    ast.Div: ("division", "__div__", "__rdiv__"),
    ast.Mod: ("modulo", "__mod__", "__rmod__"),
    ast.Pow: ("exponentiation", "__pow__", "__rpow__"),
    ast.LShift: ("left-shifting", "__lshift__", "__rlshift__"),
    ast.RShift: ("right-shifting", "__rshift__", "__rrshift__"),
    ast.BitOr: ("bitwise OR", "__or__", "__ror__"),
    ast.BitXor: ("bitwise XOR", "__xor__", "__rxor__"),
    ast.BitAnd: ("bitwise AND", "__and__", "__rand__"),
    ast.FloorDiv: ("floor division", "__floordiv__", "__rfloordiv__"),
    ast.Invert: ("inversion", "__invert__", None),
    ast.UAdd: ("unary positive", "__pos__", None),
    ast.USub: ("unary negation", "__neg__", None),
}


# these don't appear to be in the standard types module
SlotWrapperType = type(type.__init__)
MethodDescriptorType = type(list.append)

FunctionInfo = collections.namedtuple(
    "FunctionInfo",
    [
        "async_kind",  # AsyncFunctionKind
        "is_classmethod",  # has @classmethod
        "is_staticmethod",  # has @staticmethod
        "is_decorated_coroutine",  # has @asyncio.coroutine
        # a list of pairs of (decorator function, applied decorator function). These are different
        # for decorators that take arguments, like @asynq(): the first element will be the asynq
        # function and the second will be the result of calling asynq().
        "decorators",
    ],
)
# FunctionInfo for a vanilla function (e.g. a lambda)
_DEFAULT_FUNCTION_INFO = FunctionInfo(AsyncFunctionKind.normal, False, False, False, [])
_BOOL_DUNDER = "__bool__" if six.PY3 else "__nonzero__"


class ClassAttributeChecker(object):
    """Helper class to keep track of attributes that are read and set on instances."""

    def __init__(self, config, enabled=True, should_check_unused_attributes=False):
        # we might not have examined all parent classes when looking for attributes set
        # we dump them here. incase the callers want to extend coverage.
        self.unexamined_base_classes = set()
        self.modules_examined = set()
        self.enabled = enabled
        self.should_check_unused_attributes = should_check_unused_attributes
        self.all_failures = []
        self.types_with_dynamic_attrs = set()
        self.config = config
        self.filename_to_visitor = {}
        # Dictionary from type to list of (attr_name, node, filename) tuples
        self.attributes_read = collections.defaultdict(list)
        # Dictionary from type to set of attributes that are set on that class
        self.attributes_set = collections.defaultdict(set)
        # Used for attribute value inference
        self.attribute_values = collections.defaultdict(dict)
        # Classes that we have examined the AST for
        self.classes_examined = {
            self.serialize_type(typ)
            for typ in config.IGNORED_TYPES_FOR_ATTRIBUTE_CHECKING
        }

    def __enter__(self):
        if self.enabled:
            return self
        else:
            return None

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is None and self.enabled:
            self.check_attribute_reads()

            if self.should_check_unused_attributes:
                self.check_unused_attributes()

    def record_attribute_read(self, typ, attr_name, node, visitor):
        """Records that attribute attr_name was accessed on type typ."""
        self.filename_to_visitor[visitor.filename] = visitor
        serialized = self.serialize_type(typ)
        if serialized is not None:
            self.attributes_read[serialized].append((attr_name, node, visitor.filename))

    def record_attribute_set(self, typ, attr_name, node, value):
        """Records that attribute attr_name was set on type typ."""
        serialized = self.serialize_type(typ)
        if serialized is None:
            return
        self.attributes_set[serialized].add(attr_name)
        self.merge_attribute_value(serialized, attr_name, value)

    def merge_attribute_value(self, serialized, attr_name, value):
        try:
            pickle.loads(pickle.dumps(value))
        except Exception:
            # If we can't serialize it, don't attempt to store it.
            value = UNRESOLVED_VALUE
        scope = self.attribute_values[serialized]
        if attr_name not in scope:
            scope[attr_name] = value
        elif scope[attr_name] == value:
            pass
        else:
            scope[attr_name] = unite_values(scope[attr_name], value)

    def record_type_has_dynamic_attrs(self, typ):
        serialized = self.serialize_type(typ)
        if serialized is not None:
            self.types_with_dynamic_attrs.add(serialized)

    def record_class_examined(self, cls):
        """Records that we examined the attributes of class cls."""
        serialized = self.serialize_type(cls)
        if serialized is not None:
            self.classes_examined.add(serialized)

    def record_module_examined(self, module_name):
        self.modules_examined.add(module_name)

    def serialize_type(self, typ):
        """Serialize a type so it is pickleable.

        We do this to make it possible to pass ClassAttributeChecker objects around
        to parallel workers.

        """
        if isinstance(typ, super):
            typ = typ.__self_class__
        if isinstance(
            _safe_getattr(typ, "__module__", None), six.string_types
        ) and isinstance(_safe_getattr(typ, "__name__", None), six.string_types):
            module = typ.__module__
            name = typ.__name__
            if module not in sys.modules:
                return None
            if (
                self.config.unwrap_cls(_safe_getattr(sys.modules[module], name, None))
                is typ
            ):
                return (module, name)
        return None

    def unserialize_type(self, serialized):
        module, name = serialized
        if module not in sys.modules:
            __import__(module)
        try:
            return self.config.unwrap_cls(getattr(sys.modules[module], name))
        except AttributeError:
            # We've seen this happen when we import different modules under the same name.
            return None

    def get_attribute_value(self, typ, attr_name):
        """Gets the current recorded value of the attribute."""
        for base_typ in self._get_mro(typ):
            serialized_base = self.serialize_type(base_typ)
            if serialized_base is None:
                continue
            value = self.attribute_values[serialized_base].get(attr_name)
            if value is not None:
                return value
        else:
            return UNRESOLVED_VALUE

    def check_attribute_reads(self):
        """Checks that all recorded attribute reads refer to valid attributes.

        This is done by checking for each read whether the class has the attribute or whether any
        code sets the attribute on a class instance, among other conditions.

        """
        for serialized, attrs_read in sorted(
            six.iteritems(self.attributes_read), key=self._cls_sort
        ):
            typ = self.unserialize_type(serialized)
            if typ is None:
                continue
            # we setattr on it with an unresolved value, so we don't know what attributes this may
            # have
            if any(
                self.serialize_type(base_cls) in self.types_with_dynamic_attrs
                for base_cls in self._get_mro(typ)
            ):
                continue

            for attr_name, node, filename in sorted(
                attrs_read, key=lambda data: data[0]
            ):
                self._check_attribute_read(
                    typ, attr_name, node, self.filename_to_visitor[filename]
                )

    def check_unused_attributes(self):
        """Attempts to find attributes.

        This relies on comparing the set of attributes read on each class with the attributes in the
        class's __dict__. It has many false positives and should be considered experimental.

        Some known causes of false positives:
        - Methods called in base classes of children (mixins)
        - Special methods like __eq__
        - Insufficiently powerful type inference

        """
        all_attrs_read = collections.defaultdict(set)

        def _add_attrs(typ, attr_names_read):
            if typ is None:
                return
            all_attrs_read[typ] |= attr_names_read
            for base_cls in typ.__bases__:
                all_attrs_read[base_cls] |= attr_names_read
            if isinstance(typ, type):
                for child_cls in qcore.inspection.get_subclass_tree(typ):
                    all_attrs_read[child_cls] |= attr_names_read

        for serialized, attrs_read in six.iteritems(self.attributes_read):
            attr_names_read = {attr_name for attr_name, _, _ in attrs_read}
            _add_attrs(self.unserialize_type(serialized), attr_names_read)

        for typ, attrs in self.config.IGNORED_UNUSED_ATTRS_BY_CLASS:
            _add_attrs(typ, attrs)

        used_bases = tuple(self.config.USED_BASE_CLASSES)

        for typ, attrs_read in sorted(
            six.iteritems(all_attrs_read), key=self._cls_sort
        ):
            if self.serialize_type(typ) not in self.classes_examined or issubclass(
                typ, used_bases
            ):
                continue
            existing_attrs = set(typ.__dict__.keys())
            for attr in existing_attrs - attrs_read - self.config.IGNORED_UNUSED_ATTRS:
                # server calls will always show up as unused here
                if _safe_getattr(_safe_getattr(typ, attr, None), "server_call", False):
                    continue
                print("Unused method: %r.%s" % (typ, attr))

    # sort by module + name in order to get errors in a reasonable order
    def _cls_sort(self, pair):
        typ = pair[0]
        if hasattr(typ, "__name__") and isinstance(typ.__name__, six.string_types):
            return (typ.__module__, typ.__name__)
        else:
            return (six.text_type(typ), "")

    def _check_attribute_read(self, typ, attr_name, node, visitor):
        # class itself has the attribute
        if hasattr(typ, attr_name):
            return
        # the attribute is in __annotations__, e.g. a dataclass
        if _has_annotation_for_attr(typ, attr_name) or _get_attrs_attribute(
            typ, attr_name
        ):
            return

        # name mangling
        if (
            attr_name.startswith("__")
            and hasattr(typ, "__name__")
            and hasattr(typ, "_%s%s" % (typ.__name__, attr_name))
        ):
            return

        # can't be sure whether it exists if class has __getattr__
        if hasattr(typ, "__getattr__") or (
            hasattr(typ, "__getattribute__")
            and typ.__getattribute__ is not object.__getattribute__
        ):
            return

        # instances of old-style classes have __class__ available, even though the class doesn't
        if attr_name == "__class__":
            return

        serialized = self.serialize_type(typ)

        # it was set on an instance of the class
        if attr_name in self.attributes_set[serialized]:
            return

        # web browser test classes
        if attr_name == "browser" and hasattr(typ, "_pre_setup"):
            return

        base_classes_examined = {typ}
        any_base_classes_unexamined = False
        for base_cls in self._get_mro(typ):
            # the attribute is in __annotations__, e.g. a dataclass
            if _has_annotation_for_attr(base_cls, attr_name):
                return

            if self._should_reject_unexamined(base_cls):
                self.unexamined_base_classes.add(base_cls)
                any_base_classes_unexamined = True
                continue

            # attribute was set on the base class
            if attr_name in self.attributes_set[
                self.serialize_type(base_cls)
            ] or hasattr(base_cls, attr_name):
                return

            base_classes_examined.add(base_cls)

        if any_base_classes_unexamined:
            return

        if not isinstance(typ, type):
            # old-style class; don't want to support
            return

        # if it's on a child class it's also ok
        for child_cls in qcore.inspection.get_subclass_tree(typ):
            # also check the child classes' base classes, because mixins sometimes use attributes
            # defined on other parents of their child classes
            for base_cls in self._get_mro(child_cls):
                if base_cls in base_classes_examined:
                    continue

                if attr_name in self.attributes_set[
                    self.serialize_type(base_cls)
                ] or hasattr(base_cls, attr_name):
                    return

                if self._should_reject_unexamined(base_cls):
                    visitor.log(
                        logging.INFO,
                        "Rejecting because of unexamined child base class",
                        (typ, base_cls, attr_name),
                    )
                    return

                base_classes_examined.add(base_cls)

        message = visitor.show_error(
            node,
            "Attribute %s of type %s probably does not exist" % (attr_name, typ),
            ErrorCode.attribute_is_never_set,
        )
        # message can be None if the error is intercepted by error code settings or ignore
        # directives
        if message is not None:
            self.all_failures.append(message)

    def _should_reject_unexamined(self, base_cls):
        """Whether an undefined attribute should be ignored because base_cls was not examined.

        This is to keep the script from concluding that an attribute does not exist because it was
        defined on a base class whose AST was not examined.

        In two cases we still want to throw an error for undefined components even if a base class
        was not examined:
        - If the base class's module was examined, it is probably a wrapper class created by a
          decorator that does not set additional attributes.
        - If the base class is a Cython class, it should not set any attributes that are not defined
          on the class.

        """
        result = (
            self.serialize_type(base_cls) not in self.classes_examined
            and base_cls.__module__ not in self.modules_examined
            and not qcore.inspection.is_cython_class(base_cls)
        )
        if not result:
            self.unexamined_base_classes.add(base_cls)
        return result

    def _get_mro(self, typ):
        if isinstance(typ, super):
            typ_for_mro = typ.__thisclass__
        else:
            typ_for_mro = typ
        try:
            return inspect.getmro(typ_for_mro)
        except AttributeError:
            # It's not actually a class.
            return []


class StackedContexts(object):
    """Object to keep track of a stack of states.

    This is used to indicate all the AST node types that are parents of the node being examined.

    """

    def __init__(self):
        self.contexts = []

    def includes(self, typ):
        return any(isinstance(val, typ) for val in self.contexts)

    def nth_parent(self, n):
        return self.contexts[-n] if len(self.contexts) >= n else None

    def nearest_enclosing(self, typ):
        for node in reversed(self.contexts):
            if isinstance(node, typ):
                return node
        else:
            return None

    @contextlib.contextmanager
    def add(self, value):
        """Context manager to add a context to the stack."""
        self.contexts.append(value)
        try:
            yield
        finally:
            self.contexts.pop()


class CallSiteCollector(object):
    """Class to record function calls with their origin."""

    def __init__(self):
        self.map = collections.defaultdict(list)

    def record_call(self, caller, callee):
        try:
            self.map[callee].append(caller)
        except TypeError:
            # Unhashable callee. This is mostly calls to bound versions of list.append. We could get
            # the unbound method, but that doesn't seem very useful, so we just ignore it.
            pass


class NameCheckVisitor(node_visitor.ReplacingNodeVisitor):
    """Visitor class that infers the type and value of Python objects and detects some errors."""

    error_code_enum = ErrorCode
    config = Config()  # subclasses may override this with a more specific config

    def __init__(
        self,
        filename,
        contents,
        tree,
        settings=None,
        fail_after_first=False,
        verbosity=logging.CRITICAL,
        unused_finder=None,
        module=None,
        attribute_checker=None,
        arg_spec_cache=None,
        collector=None,
        annotate=False,
        add_ignores=False,
    ):
        super(NameCheckVisitor, self).__init__(
            filename,
            contents,
            tree,
            settings,
            fail_after_first=fail_after_first,
            verbosity=verbosity,
            add_ignores=add_ignores,
        )

        # State (to use in with qcore.override)
        self.state = None
        # value currently being assigned
        self.being_assigned = UNRESOLVED_VALUE
        # current class (for inferring the type of cls and self arguments)
        self.current_class = None
        self.current_function_name = None

        # async
        self.async_kind = AsyncFunctionKind.non_async
        self.is_generator = False  # set to True if this function is a generator
        # if true, we annotate each node we visit with its inferred value
        self.annotate = annotate
        # true if we're in the body of a comprehension's loop
        self.in_comprehension_body = False

        if module is not None:
            self.module = module
            self.is_compiled = False
        else:
            self.module, self.is_compiled = self._load_module()

        # Data storage objects
        self.unused_finder = unused_finder
        self.attribute_checker = attribute_checker
        if arg_spec_cache is None:
            arg_spec_cache = ArgSpecCache(self.config)
        self.arg_spec_cache = arg_spec_cache
        if (
            self.attribute_checker is not None
            and self.module is not None
            and not self.is_compiled
        ):
            self.attribute_checker.record_module_examined(self.module.__name__)
        self.scope = StackedScopes(self.module)
        self.node_context = StackedContexts()
        self.asynq_checker = AsynqChecker(
            self.config,
            self.module,
            self._lines,
            self.show_error,
            self.log,
            self.replace_node,
        )
        self.yield_checker = YieldChecker(self)
        self.current_function = None
        self.expected_return_value = None
        self.is_async_def = False
        self.collector = collector
        self.import_name_to_node = {}
        self.imports_added = set()
        self.future_imports = set()  # active future imports in this file
        self.return_values = []

        self._name_node_to_statement = None
        # Cache the return values of functions within this file, so that we can use them to
        # infer types. Previously, we cached this globally, but that makes things non-
        # deterministic because we'll start depending on the order modules are checked.
        self._argspec_to_retval = {}
        self._method_cache = {}
        self._fill_method_cache()

    def __reduce_ex__(self, proto):
        # Only pickle the attributes needed to get error reporting working
        return self.__class__, (self.filename, self.contents, self.tree, self.settings)

    def _load_module(self):
        """Sets the module_path and module for this file."""
        self.log(logging.INFO, "Checking file", (self.filename, os.getpid()))

        try:
            return self.load_module(self.filename)
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            # don't re-raise the error, just proceed without a module object
            # this can happen with scripts that aren't intended to be imported
            if not self.has_file_level_ignore():
                traceback.print_exc()
                if self.tree.body:
                    node = self.tree.body[0]
                else:
                    node = None
                self.show_error(
                    node,
                    "Failed to import {} due to {!r}".format(self.filename, e),
                    error_code=ErrorCode.import_failed,
                )
            return None, False

    def load_module(self, filename):
        return importer.load_module_from_file(
            filename, self.config.PATHS_EXCLUDED_FROM_IMPORT
        )

    def check(self):
        """Runs the visitor on this module."""
        try:
            if self.is_compiled:
                # skip compiled (Cythonized) files because pyanalyze will misinterpret the
                # AST in some cases (for example, if a function was cdefed)
                return []
            if self.module is None:
                # If we could not import the module, other checks frequently fail.
                return self.all_failures
            with qcore.override(self, "state", VisitorState.collect_names):
                self.visit(self.tree)
            with qcore.override(self, "state", VisitorState.check_names):
                self.visit(self.tree)
            # This doesn't deal correctly with errors from the attribute checker. Therefore,
            # leaving this check disabled by default for now.
            self.show_errors_for_unused_ignores(ErrorCode.unused_ignore)
            self.show_errors_for_bare_ignores(ErrorCode.bare_ignore)
        except node_visitor.VisitorError:
            raise
        except Exception as e:
            self.show_error(
                None,
                "%s\nInternal error: %r" % (traceback.format_exc(), e),
                error_code=ErrorCode.internal_error,
            )
        # Recover memory used for the AST. We keep the visitor object around later in order
        # to show ClassAttributeChecker errors, but those don't need the full AST.
        self.tree = None
        self._lines.__cached_per_instance_cache__.clear()
        self._argspec_to_retval.clear()
        return self.all_failures

    def visit(self, node):
        """Visits a node and ensures that it returns UNRESOLVED_VALUE when necessary."""
        # inline self.node_context.add and the superclass's visit() for performance
        method = self._method_cache[type(node)]
        self.node_context.contexts.append(node)
        try:
            # This part inlines ReplacingNodeVisitor.visit
            if isinstance(node, ast.stmt):
                # inline qcore.override here
                old_statement = self.current_statement
                try:
                    self.current_statement = node
                    ret = method(node)
                finally:
                    self.current_statement = old_statement
            else:
                ret = method(node)
        except node_visitor.VisitorError:
            raise
        except Exception as e:
            self.show_error(
                node,
                "%s\nInternal error: %r" % (traceback.format_exc(), e),
                error_code=ErrorCode.internal_error,
            )
            ret = UNRESOLVED_VALUE
        finally:
            self.node_context.contexts.pop()
        if ret is None:
            ret = UNRESOLVED_VALUE
        if self.annotate:
            node.inferred_value = ret
        return ret

    def generic_visit(self, node):
        """Inlined version of ast.Visitor.generic_visit for performance."""
        for field in node._fields:
            try:
                value = getattr(node, field)
            except AttributeError:
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def _fill_method_cache(self):
        for typ in qcore.inspection.get_subclass_tree(ast.AST):
            method = "visit_" + typ.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[typ] = visitor

    def _is_collecting(self):
        return self.state == VisitorState.collect_names

    def _is_checking(self):
        return self.state == VisitorState.check_names

    def _show_error_if_checking(
        self, node, msg=None, error_code=None, replacement=None
    ):
        """We usually should show errors only in the check_names state to avoid duplicate errors."""
        if self._is_checking():
            self.show_error(node, msg, error_code=error_code, replacement=replacement)

    def _set_name_in_scope(self, varname, node, value=UNRESOLVED_VALUE):
        scope_type = self.scope.scope_type()
        if not isinstance(value, KnownValue) and scope_type == ScopeType.module_scope:
            try:
                value = KnownValue(getattr(self.module, varname))
            except AttributeError:
                pass
        if scope_type == ScopeType.class_scope:
            self._check_for_class_variable_redefinition(varname, node)
        self.scope.set(varname, value, node, self.state)

    def _check_for_class_variable_redefinition(self, varname, node):
        if varname not in self.scope.current_scope():
            return

        # exclude cases where we do @<property>.setter
        # use __dict__ rather than getattr because properties override __get__
        if self.current_class is not None and isinstance(
            self.current_class.__dict__.get(varname), property
        ):
            return

        # allow augmenting an attribute
        if isinstance(self.current_statement, ast.AugAssign):
            return

        self.show_error(
            node,
            "Name {} is already defined".format(varname),
            error_code=ErrorCode.class_variable_redefinition,
        )

    def resolve_name(self, node, error_node=None, suppress_errors=False):
        """Resolves a Name node to a value.

        If error_node is given, it is used (instead of Node) to show errors on.

        """
        if error_node is None:
            error_node = node
        value = self.scope.get(node.id, node, self.state)
        if value is UNINITIALIZED_VALUE:
            if suppress_errors or node.id in self.config.IGNORED_VARIABLES:
                self.log(logging.INFO, "ignoring undefined name", node.id)
            else:
                self._maybe_show_missing_import_error(node)
                self._show_error_if_checking(
                    error_node,
                    "Undefined name: %s" % (node.id,),
                    ErrorCode.undefined_name,
                )
            return UNRESOLVED_VALUE
        if isinstance(value, MultiValuedValue):
            if any(subval is UNINITIALIZED_VALUE for subval in value.vals):
                self._show_error_if_checking(
                    error_node,
                    "%s may be used uninitialized" % node.id,
                    ErrorCode.possibly_undefined_name,
                )
                return MultiValuedValue(
                    [
                        UNRESOLVED_VALUE if subval is UNINITIALIZED_VALUE else subval
                        for subval in value.vals
                    ]
                )
        return value

    def _maybe_show_missing_import_error(self, node):
        """Shows errors that suggest adding an import statement in the semi-right place.

        This mostly exists to make refactorings that add imported names easier. A couple of
        potential improvements:
        - Currently it just adds new imports before the first existing one. It could be made
          smarter.
        - It doesn't know what to do if the file doesn't have any imports.
        - The code adding more entries to a from ... import will fail if the existing node spans
          multiple lines.

        """
        if not self._is_checking():
            return
        if node.id not in self.config.NAMES_TO_IMPORTS:
            return
        if node.id in self.imports_added:
            return
        self.imports_added.add(node.id)
        target = self.config.NAMES_TO_IMPORTS[node.id]
        if target is None or target not in self.import_name_to_node:
            # add the import
            try:
                target_node = self._get_first_import_node()
            except ValueError:
                return  # no import nodes, you're on your own
            lineno = target_node.lineno
            if target is None:
                new_line = "import %s\n" % (node.id,)
            else:
                new_line = "from %s import %s\n" % (target, node.id)
            new_lines = [new_line, self._lines()[lineno - 1]]
            self._show_error_if_checking(
                target_node,
                "add an import for %s" % (node.id,),
                error_code=ErrorCode.add_import,
                replacement=node_visitor.Replacement([lineno], new_lines),
            )
        else:
            existing = self.import_name_to_node[target]
            names = existing.names + [ast.alias(name=node.id, asname=None)]
            names = sorted(names, key=lambda alias: alias.name)
            existing.names = (
                names  # so that when we add more names this one is maintained
            )
            new_node = ast.ImportFrom(
                module=existing.module, names=names, level=existing.level
            )
            new_code = decompile(new_node)
            self._show_error_if_checking(
                existing,
                "add an import for %s" % (node.id,),
                error_code=ErrorCode.add_import,
                replacement=node_visitor.Replacement([existing.lineno], [new_code]),
            )

    def _get_first_import_node(self):
        return min(self.import_name_to_node.values(), key=lambda node: node.lineno)

    def _generic_visit_list(self, lst):
        return [self.visit(node) for node in lst]

    def _is_write_ctx(self, ctx):
        return isinstance(ctx, (ast.Store, ast.Param))

    def _is_read_ctx(self, ctx):
        return isinstance(ctx, (ast.Load, ast.Del))

    @contextlib.contextmanager
    def _set_current_class(self, current_class):
        with qcore.override(self, "current_class", current_class), qcore.override(
            self.asynq_checker, "current_class", current_class
        ):
            yield

    def visit_ClassDef(self, node):
        self._generic_visit_list(node.decorator_list)
        self._generic_visit_list(node.bases)
        value = self._visit_class_and_get_value(node)
        self._set_name_in_scope(node.name, node, value)
        return value

    def _visit_class_and_get_value(self, node):
        if self._is_checking():
            if self.scope.scope_type() == ScopeType.module_scope:
                cls_obj = self.scope.get(node.name, node, self.state)
            else:
                cls_obj = UNRESOLVED_VALUE

            if isinstance(cls_obj, MultiValuedValue) and self.module is not None:
                # if there are multiple, see if there is only one that matches this module
                possible_values = [
                    val
                    for val in cls_obj.vals
                    if isinstance(val, KnownValue)
                    and isinstance(val.val, type)
                    and _safe_getattr(val.val, "__module__", None)
                    == self.module.__name__
                ]
                if len(possible_values) == 1:
                    cls_obj = possible_values[0]

            if isinstance(cls_obj, KnownValue):
                cls_obj = KnownValue(self.config.unwrap_cls(cls_obj.val))
                current_class = cls_obj.val
                self._record_class_examined(current_class)
            else:
                current_class = None

            with self.scope.add_scope(
                ScopeType.class_scope, scope_node=None
            ), self._set_current_class(current_class):
                self._generic_visit_list(node.body)

            if isinstance(cls_obj, KnownValue):
                return cls_obj

        return TypedValue(type)

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node, is_coroutine=True)

    def visit_FunctionDef(self, node, is_coroutine=False):
        with qcore.override(self, "current_class", None):
            info = self._visit_decorators_and_check_asynq(node.decorator_list)
        defaults, kw_defaults = self._visit_defaults(node)

        scope_type = self.scope.scope_type()
        if scope_type == ScopeType.module_scope and self.module is not None:
            potential_function = _safe_getattr(self.module, node.name, None)
        elif scope_type == ScopeType.class_scope and self.current_class is not None:
            potential_function = _safe_getattr(self.current_class, node.name, None)
        else:
            potential_function = None

        self.yield_checker.reset_yield_checks()

        # This code handles nested functions
        evaled_function = None
        if potential_function is None:
            if scope_type != ScopeType.function_scope:
                self.log(
                    logging.INFO, "Failed to find function", (node.name, scope_type)
                )
            evaled_function = self._get_evaled_function(node, info.decorators)
            # evaled_function should be a KnownValue of a function unless a decorator messed it up.
            if isinstance(evaled_function, KnownValue):
                potential_function = evaled_function.val

        if hasattr(node, "returns") and node.returns is not None:
            return_annotation = self.visit(node.returns)
            expected_return_value = self._value_of_annotation_type(
                return_annotation, node.returns
            )
        else:
            expected_return_value = None

        if evaled_function is not None:
            self._set_name_in_scope(node.name, node, evaled_function)
        else:
            self._set_name_in_scope(node.name, node, KnownValue(potential_function))

        with self.asynq_checker.set_func_name(
            node.name, async_kind=info.async_kind, is_classmethod=info.is_classmethod
        ), qcore.override(self, "yield_checker", YieldChecker(self)), qcore.override(
            self, "is_async_def", is_coroutine
        ), qcore.override(
            self, "current_function_name", node.name
        ), qcore.override(
            self, "current_function", potential_function
        ), qcore.override(
            self, "expected_return_value", expected_return_value
        ):
            return_value, has_return, is_generator = self._visit_function_body(
                node,
                function_info=info,
                name=node.name,
                defaults=defaults,
                kw_defaults=kw_defaults,
            )

        if (
            not has_return
            and expected_return_value is not None
            and expected_return_value != KnownValue(None)
            and expected_return_value is not NO_RETURN_VALUE
            and not any(
                decorator == KnownValue(abstractmethod)
                for _, decorator in info.decorators
            )
        ):
            self._show_error_if_checking(
                node,
                "Function may exit without returning a value",
                error_code=ErrorCode.incompatible_return_value,
            )

        if evaled_function:
            return evaled_function

        if info.async_kind == AsyncFunctionKind.normal and return_value.is_type(
            (asynq.FutureBase, asynq.AsyncTask)
        ):
            self._show_error_if_checking(node, error_code=ErrorCode.task_needs_yield)

        # If there was an annotation, use it as the return value in the
        # _argspec_to_retval cache, even if we inferred something else while visiting
        # the function.
        if not is_generator and expected_return_value is not None:
            return_value = expected_return_value

        if is_generator and return_value == KnownValue(None):
            return_value = UNRESOLVED_VALUE

        # pure async functions are otherwise incorrectly inferred as returning whatever the
        # underlying function returns
        if info.async_kind == AsyncFunctionKind.pure:
            return_value = AsyncTaskIncompleteValue(
                _get_task_cls(potential_function), return_value
            )

        # If a decorator turned the function async, believe it. This fixes one
        # instance of the general problem where a decorator can change the return
        # type, but we still look at the original function's annotation. A more
        # principled fix for such issues will have to wait for better support for
        # callable types.
        if potential_function is not None and not is_coroutine:
            is_coroutine = _is_coroutine_function(potential_function)

        if is_coroutine or info.is_decorated_coroutine:
            return_value = AwaitableIncompleteValue(return_value)

        try:
            argspec = self.arg_spec_cache.get_argspec(
                potential_function, name=node.name, logger=self.log
            )
        except TypeError:
            return KnownValue(potential_function)
        if argspec is not None:
            if info.async_kind != AsyncFunctionKind.async_proxy:
                # don't attempt to infer the return value of async_proxy functions, since it will be
                # set within the Future returned
                # without this, we'll incorrectly infer the return value to be the Future instead of
                # the Future's value
                self._argspec_to_retval[id(argspec)] = return_value
        else:
            self.log(logging.DEBUG, "No argspec", (potential_function, node))
        return KnownValue(potential_function)

    def _visit_defaults(self, node):
        with qcore.override(self, "current_class", None):
            defaults = self._generic_visit_list(node.args.defaults)

            if hasattr(node.args, "kw_defaults"):  # py3
                kw_defaults = [
                    None if kw_default is None else self.visit(kw_default)
                    for kw_default in node.args.kw_defaults
                ]
            else:
                kw_defaults = None
            return defaults, kw_defaults

    def is_value_compatible(self, val1, val2):
        try:
            return val1.is_value_compatible(val2)
        except Exception:
            # is_value_compatible() can call into some user code
            # (e.g. __eq__ and __subclasscheck__), so ignore any errors
            self.log(
                logging.DEBUG, "is_value_compatible failed", traceback.format_exc()
            )
            return True

    def _get_evaled_function(self, node, decorators):
        to_apply = []
        for decorator, applied_decorator in decorators:
            if (
                not isinstance(decorator, KnownValue)
                or not isinstance(applied_decorator, KnownValue)
                or decorator.val not in self.config.SAFE_DECORATORS_FOR_NESTED_FUNCTIONS
            ):
                self.log(
                    logging.DEBUG,
                    "Reject nested function because of decorator",
                    (node, decorator),
                )
                return TypedValue(types.FunctionType)
            else:
                to_apply.append(applied_decorator.val)
        scope = {}
        kwargs = {
            "args": [self._strip_annotation(arg) for arg in node.args.args],
            "vararg": self._strip_annotation(node.args.vararg),
            "kwarg": self._strip_annotation(node.args.kwarg),
            "defaults": [ast.Name(id="None") for _ in node.args.defaults],
        }
        if hasattr(node.args, "kwonlyargs"):
            kwargs["kwonlyargs"] = [
                self._strip_annotation(arg) for arg in node.args.kwonlyargs
            ]
            kwargs["kw_defaults"] = [ast.Name(id="None") for _ in node.args.kw_defaults]
        new_args = ast.arguments(**kwargs)
        new_node = ast.FunctionDef(
            name=node.name, args=new_args, body=[ast.Pass()], decorator_list=[]
        )
        code = decompile(new_node)
        exec (code, scope, scope)
        fn = scope[node.name]
        for decorator in reversed(to_apply):
            fn = decorator(fn)
        try:
            fn._pyanalyze_is_nested_function = True
            fn._pyanalyze_parent_function = self.current_function
        except AttributeError:
            # could happen if decorator wrapped it in an object that doesn't allow attribute
            # assignment
            pass
        return KnownValue(fn)

    def _strip_annotation(self, node):
        if hasattr(node, "annotation"):  # py3
            return ast.arg(arg=node.arg, annotation=None)
        else:
            return node

    def visit_Lambda(self, node):
        defaults, kw_defaults = self._visit_defaults(node)

        with self.asynq_checker.set_func_name("<lambda>"):
            self._visit_function_body(node, defaults=defaults, kw_defaults=kw_defaults)

    def _visit_decorators_and_check_asynq(self, decorator_list):
        """Visits a function's decorator list. Returns a FunctionInfo namedtuple."""
        async_kind = AsyncFunctionKind.non_async
        is_classmethod = False
        is_decorated_coroutine = False
        is_staticmethod = False
        decorators = []
        for decorator in decorator_list:
            # We have to descend into the Call node because the result of
            # asynq.asynq() is a one-off function that we can't test against.
            # This means that the decorator will be visited more than once, which seems OK.
            if isinstance(decorator, ast.Call):
                decorator_value = self.visit(decorator)
                callee = self.visit(decorator.func)
                if isinstance(callee, KnownValue):
                    if self._safe_in(callee.val, self.config.ASYNQ_DECORATORS):
                        if any(kw.arg == "pure" for kw in decorator.keywords):
                            async_kind = AsyncFunctionKind.pure
                        else:
                            async_kind = AsyncFunctionKind.normal
                    elif self._safe_in(callee.val, self.config.ASYNC_PROXY_DECORATORS):
                        # @async_proxy(pure=True) is a noop, so don't treat it specially
                        if not any(kw.arg == "pure" for kw in decorator.keywords):
                            async_kind = AsyncFunctionKind.async_proxy
                decorators.append((callee, decorator_value))
            else:
                decorator_value = self.visit(decorator)
                if decorator_value == KnownValue(classmethod):
                    is_classmethod = True
                elif decorator_value == KnownValue(staticmethod):
                    is_staticmethod = True
                elif asyncio is not None and decorator_value == KnownValue(
                    asyncio.coroutine
                ):
                    is_decorated_coroutine = True
                decorators.append((decorator_value, decorator_value))
        return FunctionInfo(
            async_kind=async_kind,
            is_decorated_coroutine=is_decorated_coroutine,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            decorators=decorators,
        )

    def _visit_function_body(
        self,
        node,
        function_info=_DEFAULT_FUNCTION_INFO,
        name=None,
        defaults=None,
        kw_defaults=None,
    ):
        is_collecting = self._is_collecting()
        if is_collecting and not self.scope.contains_scope_of_type(
            ScopeType.function_scope
        ):
            return UNRESOLVED_VALUE, False, False

        # We pass in the node to add_scope() and visit the body once in collecting
        # mode if in a nested function, so that constraints on nonlocals in the outer
        # scope propagate into this scope. This means that we'll use the constraints
        # of the place where the function is defined, not those of where the function
        # is called, which is strictly speaking wrong but should be fine in practice.
        with self.scope.add_scope(
            ScopeType.function_scope, scope_node=node
        ), qcore.override(self, "is_generator", False), qcore.override(
            self, "async_kind", function_info.async_kind
        ), qcore.override(
            self, "_name_node_to_statement", {}
        ):
            scope = self.scope.current_scope()

            if isinstance(node.body, list):
                body = node.body
            else:
                # hack for lambdas
                body = [node.body]

            class_ctx = (
                qcore.empty_context
                if not self.scope.is_nested_function()
                else qcore.override(self, "current_class", None)
            )
            with class_ctx:
                args = self._visit_function_args(
                    node, function_info, defaults, kw_defaults
                )

            with qcore.override(
                self, "state", VisitorState.collect_names
            ), qcore.override(self, "return_values", []):
                self._generic_visit_list(body)
            if is_collecting:
                return UNRESOLVED_VALUE, False, self.is_generator

            # otherwise we may end up using results from the last yield (generated during the
            # collect state) to evaluate the first one visited during the check state
            self.yield_checker.reset_yield_checks()

            with qcore.override(self, "current_class", None), qcore.override(
                self, "state", VisitorState.check_names
            ), qcore.override(self, "return_values", []):
                self._generic_visit_list(body)
                return_values = self.return_values
                return_set = scope.get_local(LEAVES_SCOPE, None, self.state)

            self._check_function_unused_vars(scope, args)
            return self._compute_return_type(node, name, return_values, return_set)

    def _compute_return_type(self, node, name, return_values, return_set):
        # Ignore generators for now.
        if return_set is UNRESOLVED_VALUE or self.is_generator:
            has_return = True
        elif return_set is UNINITIALIZED_VALUE:
            has_return = False
        else:
            assert False, return_set
        # if the return value was never set, the function returns None
        if not return_values:
            if name is not None:
                method_return_type.check_no_return(node, self, name)
            return KnownValue(None), has_return, self.is_generator
        # None is added to return_values if the function raises an error.
        return_values = [val for val in return_values if val is not None]
        # If it only ever raises an error, we don't know what it returns. Strictly
        # this should perhaps be NoReturnValue, but that leads to issues because
        # in practice this condition often occurs in abstract methods that just
        # raise NotImplementedError.
        if not return_values:
            return UNRESOLVED_VALUE, has_return, self.is_generator
        else:
            return unite_values(*return_values), has_return, self.is_generator

    def _check_function_unused_vars(self, scope, args, enclosing_statement=None):
        """Shows errors for any unused variables in the function."""
        all_def_nodes = set(
            chain.from_iterable(scope.name_to_all_definition_nodes.values())
        )
        all_used_def_nodes = set(
            chain.from_iterable(scope.usage_to_definition_nodes.values())
        )
        arg_nodes = set(args)
        all_unused_nodes = all_def_nodes - all_used_def_nodes
        for unused in all_unused_nodes:
            # Ignore names not defined through a Name node (e.g., some function arguments)
            if not isinstance(unused, ast.Name):
                continue
            # Ignore names that are meant to be ignored
            if unused.id.startswith("_"):
                continue
            # Ignore arguments
            if unused in arg_nodes:
                continue
            # Ignore names involved in global and similar declarations
            if unused.id in scope.accessed_from_special_nodes:
                continue
            replacement = None
            if self._name_node_to_statement is not None:
                # Ignore some names defined in unpacking assignments. This should behave as follows:
                #   a, b = c()  # error only if a and b are both unused
                #   a, b = yield c.asynq()  # same
                #   a, b = yield (func1.asynq(), func2.asynq())  # error if either a or b is unused
                #   [None for i in range(3)]  # error
                #   [a for a, b in pairs]  # no error
                #   [None for a, b in pairs]  # error
                statement = self._name_node_to_statement[unused]
                if isinstance(statement, ast.Assign):
                    # it's an assignment
                    if not (
                        isinstance(statement.value, ast.Yield)
                        and isinstance(statement.value.value, ast.Tuple)
                    ):
                        # but not an assignment originating from yielding a tuple (which is probably an
                        # async yield)

                        # We need to loop over the targets to handle code like "a, b = c = func()". If
                        # the target containing our unused variable is a tuple and some of its members
                        # are not unused, ignore it.
                        partly_used_target = False
                        for target in statement.targets:
                            if (
                                isinstance(target, (ast.List, ast.Tuple))
                                and _contains_node(target.elts, unused)
                                and not _all_names_unused(target.elts, all_unused_nodes)
                            ):
                                partly_used_target = True
                                break
                        if partly_used_target:
                            continue
                    if len(statement.targets) == 1 and not isinstance(
                        statement.targets[0], (ast.List, ast.Tuple)
                    ):
                        replacement = self.remove_node(unused, statement)
                elif isinstance(statement, ast.comprehension):
                    if isinstance(statement.target, ast.Tuple):
                        if not _all_names_unused(
                            statement.target.elts, all_unused_nodes
                        ):
                            continue
                    else:
                        replacement = self.replace_node(
                            unused,
                            ast.Name(id="_", ctx=ast.Store()),
                            enclosing_statement,
                        )
                elif hasattr(ast, "AnnAssign") and isinstance(statement, ast.AnnAssign):
                    # (Python 3.6+ only) ignore assignments in AnnAssign nodes, which don't actually
                    # bind the name
                    continue
            self._show_error_if_checking(
                unused,
                "Variable {} is not read after being written to".format(unused.id),
                error_code=ErrorCode.unused_variable,
                replacement=replacement,
            )

    def _visit_function_args(self, node, function_info, defaults, kw_defaults):
        """Visits and checks the arguments to a function. Returns the list of argument names."""
        self._check_method_first_arg(node, function_info=function_info)

        num_without_defaults = len(node.args.args) - len(defaults)
        defaults = [None] * num_without_defaults + defaults
        args = node.args.args

        if hasattr(node.args, "kwonlyargs"):
            args = args + node.args.kwonlyargs
            defaults = defaults + kw_defaults

        with qcore.override(self, "state", VisitorState.check_names):
            for idx, (arg, default) in enumerate(zip(args, defaults)):
                is_self = (
                    idx == 0
                    and self.current_class is not None
                    and not function_info.is_staticmethod
                    and not isinstance(node, ast.Lambda)
                )
                if getattr(arg, "annotation", None) is not None:
                    # Evaluate annotations in the surrounding scope,
                    # not the function's scope.
                    with self.scope.ignore_topmost_scope(), qcore.override(
                        self, "state", VisitorState.collect_names
                    ):
                        annotated_type = self.visit(arg.annotation)
                    value = self._value_of_annotation_type(
                        annotated_type, arg.annotation
                    )
                    if default is not None and not self.is_value_compatible(
                        value, default
                    ):
                        self._show_error_if_checking(
                            arg,
                            "Default value for argument %s incompatible with declared type %s"
                            % (arg.arg, value),
                            error_code=ErrorCode.incompatible_default,
                        )
                elif is_self:
                    if function_info.is_classmethod or getattr(node, "name", None) in (
                        "__init_subclass__",
                        "__new__",
                    ):
                        value = SubclassValue(self.current_class)
                    else:
                        # normal method
                        value = TypedValue(self.current_class)
                elif default is not None:
                    value = unite_values(UNRESOLVED_VALUE, default)
                else:
                    value = UNRESOLVED_VALUE

                if is_self:
                    # we need this for the implementation of super()
                    self.scope.set("%first_arg", value, "%first_arg", self.state)

                with qcore.override(self, "being_assigned", value):
                    self.visit(arg)

            if node.args.vararg is not None:
                # in py3 the vararg is wrapped in an arg object
                vararg = getattr(node.args.vararg, "arg", node.args.vararg)
                self.scope.set(vararg, TypedValue(tuple), vararg, self.state)
            if node.args.kwarg is not None:
                kwarg = getattr(node.args.kwarg, "arg", node.args.kwarg)
                self.scope.set(kwarg, TypedValue(dict), kwarg, self.state)

        return args

    def _value_of_annotation_type(self, val, node):
        """Given a value encountered in a type annotation, return a type."""
        return type_from_value(val, visitor=self, node=node)

    def _check_method_first_arg(self, node, function_info=_DEFAULT_FUNCTION_INFO):
        """Makes sure the first argument to a method is self or cls."""
        if self.current_class is None:
            return
        # staticmethods have no restrictions
        if function_info.is_staticmethod:
            return
        # try to confirm that it's actually a method
        if not hasattr(node, "name") or not hasattr(self.current_class, node.name):
            return
        first_must_be = "cls" if function_info.is_classmethod else "self"

        if len(node.args.args) < 1 or len(node.args.defaults) == len(node.args.args):
            self.show_error(
                node,
                "Method must have at least one non-keyword argument",
                ErrorCode.method_first_arg,
            )
        elif not self._arg_has_name(node.args.args[0], first_must_be):
            self.show_error(
                node,
                "First argument to method should be %s" % (first_must_be,),
                ErrorCode.method_first_arg,
            )

    def _arg_has_name(self, node, name):
        if six.PY3:
            return node.arg == name
        else:
            return isinstance(node, ast.Name) and node.id == name

    def visit_Global(self, node):
        if self.scope.scope_type() != ScopeType.function_scope:
            self._show_error_if_checking(node, error_code=ErrorCode.bad_global)
            return

        for name in node.names:
            self._set_name_in_scope(
                name, node, ReferencingValue(self.scope.module_scope(), name)
            )

    def visit_Nonlocal(self, node):
        if self.scope.scope_type() != ScopeType.function_scope:
            self._show_error_if_checking(node, error_code=ErrorCode.bad_nonlocal)
            return

        for name in node.names:
            defining_scope = self.scope.get_nonlocal_scope(
                name, self.scope.current_scope()
            )
            if defining_scope is None:
                # this is a SyntaxError, so it might be impossible to reach this branch
                self._show_error_if_checking(
                    node,
                    "nonlocal name {} does not exist in any enclosing scope".format(
                        name
                    ),
                    error_code=ErrorCode.bad_nonlocal,
                )
                defining_scope = self.scope.module_scope()
            self._set_name_in_scope(name, node, ReferencingValue(defining_scope, name))

    # Imports

    def visit_Import(self, node):
        if self.scope.scope_type() == ScopeType.module_scope:
            self._handle_imports(node.names)

            for name in node.names:
                self.import_name_to_node[name.name] = node
        else:
            self._simulate_import(node)

    def visit_ImportFrom(self, node):
        # this is used to decide where to add additional imports (after the first import), so
        # exclude __future__ imports
        if (
            self.scope.scope_type() == ScopeType.module_scope
            and node.module
            and node.module != "__future__"
        ):
            self.import_name_to_node[node.module] = node
        if node.module == "__future__":
            for name in node.names:
                self.future_imports.add(name.name)

        is_star_import = len(node.names) == 1 and node.names[0].name == "*"
        if self.scope.scope_type() == ScopeType.module_scope and not is_star_import:
            self._handle_imports(node.names)
        else:
            self._simulate_import(node, is_import_from=True)

    def _simulate_import(self, node, is_import_from=False):
        """Set the names retrieved from an import node in nontrivial situations.

        For simple imports (module-global imports that are not "from ... import *"), we can just
        retrieve the imported names from the module dictionary, but this is not possible with
        import * or when the import is within a function.

        To figure out what names would be imported in these cases, we create a fake module
        consisting of just the import statement, eval it, and set all the names in its __dict__
        in the current module scope.

        """
        if self.module is None:
            self._handle_imports(node.names)
            return

        source_code = decompile(node)

        if is_import_from:
            # the split is needed for cases like "from foo.bar import baz" if foo is unimportable
            unimportable = (
                node.module is not None
                and node.module.split(".")[0] in self.config.UNIMPORTABLE_MODULES
            )
        else:
            # need the split if the code is "import foo.bar as bar" if foo is unimportable
            unimportable = any(
                name.name.split(".")[0] in self.config.UNIMPORTABLE_MODULES
                for name in node.names
            )
        if unimportable:
            self._handle_imports(node.names)
            self.log(logging.INFO, "Ignoring import node", source_code)
            return

        # create a pseudo-module and examine its dictionary to figure out what this imports
        # default to the current __file__ if necessary
        module_file = _safe_getattr(self.module, "__file__", __file__)
        random_suffix = "".join(
            random.choice(string.ascii_lowercase) for _ in range(10)
        )
        pseudo_module_file = re.sub(r"\.pyc?$", random_suffix + ".py", module_file)
        is_init = os.path.basename(module_file) in ("__init__.py", "__init__.pyc")
        if is_init:
            pseudo_module_name = self.module.__name__ + "." + random_suffix
        else:
            pseudo_module_name = self.module.__name__ + random_suffix

        # Apparently doing 'from file_in_package import *' in an __init__.py also adds
        # file_in_package to the module's scope.
        if (
            is_import_from
            and is_init
            and node.module is not None
            and "." not in node.module
        ):  # not in the package
            if node.level == 1 or (node.level == 0 and node.module not in sys.modules):
                self._set_name_in_scope(node.module, node, TypedValue(types.ModuleType))

        with tempfile.TemporaryFile() as f:
            f.write(source_code.encode("utf-8"))
            f.seek(0)
            try:
                pseudo_module = imp.load_module(
                    pseudo_module_name,
                    f,
                    pseudo_module_file,
                    (".py", "r", imp.PY_SOURCE),
                )
            except Exception:
                # sets the name of the imported module to an UnresolvedValue so we don't get further
                # errors
                self._handle_imports(node.names)
                return
            finally:
                # clean up pyc file
                try:
                    os.unlink(pseudo_module_file + "c")
                except OSError:
                    pass
                if pseudo_module_name in sys.modules:
                    del sys.modules[pseudo_module_name]

        for name, value in six.iteritems(pseudo_module.__dict__):
            if name.startswith("__") or (
                hasattr(builtins, name) and value == getattr(builtins, name)
            ):
                continue
            self._set_name_in_scope(name, (node, name), KnownValue(value))

    def _imported_names_of_nodes(self, names):
        for node in names:
            if node.asname is not None:
                yield node.asname, node
            else:
                yield node.name.split(".")[0], node

    def _handle_imports(self, names):
        for varname, node in self._imported_names_of_nodes(names):
            self._set_name_in_scope(varname, node)

    # Comprehensions

    def visit_ListComp(self, node):
        # in python 2, list comprehensions don't generate a new scope, all others do
        # therefore, they should not go through the collecting and checking phases separately
        if six.PY3:
            return self._visit_comprehension(node, list)
        for generator in node.generators:
            self.visit(generator)

        with qcore.override(self, "in_comprehension_body", True):
            member_value = self.visit(node.elt)
        if member_value is UNRESOLVED_VALUE:
            return TypedValue(list)
        else:
            return GenericValue(list, [member_value])

    def visit_SetComp(self, node):
        return self._visit_comprehension(node, set)

    def visit_DictComp(self, node):
        if self.state == VisitorState.collect_names:
            return TypedValue(dict)
        with self.scope.add_scope(ScopeType.function_scope, scope_node=node):
            for state in (VisitorState.collect_names, VisitorState.check_names):
                with qcore.override(self, "state", state):
                    for generator in node.generators:
                        self.visit(generator)
                    with qcore.override(self, "in_comprehension_body", True):
                        key = self.visit(node.key)
                        value = self.visit(node.value)
                    ret = GenericValue(dict, [key, value])
        return ret

    def visit_GeneratorExp(self, node):
        return self._visit_comprehension(node, types.GeneratorType)

    def visit_comprehension(self, node, iterable_type=None):
        if iterable_type is None:
            iterable_type = self._member_value_of_generator(node)
        with qcore.override(self, "in_comprehension_body", True):
            with qcore.override(self, "being_assigned", iterable_type):
                self.visit(node.target)
            for cond in node.ifs:
                _, constraint = self.constraint_from_condition(cond)
                self.add_constraint(cond, constraint)

    def _member_value_of_generator(self, comprehension_node):
        is_async = getattr(comprehension_node, "is_async", False)
        iterable_type, _ = self._member_value_of_iterator(
            comprehension_node.iter, is_async
        )
        return iterable_type

    def _visit_comprehension(self, node, typ, should_create_scope=True):
        # the iteree of the first generator is executed in the enclosing scope
        iterable_type = self._member_value_of_generator(node.generators[0])
        if self.state == VisitorState.collect_names:
            # Visit it once to get usage nodes for usage of nested variables. This enables
            # us to inherit constraints on nested variables.
            # Strictly speaking this is unsafe to do for generator expressions, which may
            # be evaluated at a different place in the function than where they are defined,
            # but that is unlikely to be an issue in practice.
            with self.scope.add_scope(
                ScopeType.function_scope, scope_node=node
            ), qcore.override(self, "_name_node_to_statement", {}):
                return self._visit_comprehension_inner(node, typ, iterable_type)

        with self.scope.add_scope(
            ScopeType.function_scope, scope_node=node
        ), qcore.override(self, "_name_node_to_statement", {}):
            scope = self.scope.current_scope()
            for state in (VisitorState.collect_names, VisitorState.check_names):
                with qcore.override(self, "state", state):
                    ret = self._visit_comprehension_inner(node, typ, iterable_type)
            self._check_function_unused_vars(
                scope,
                (),
                enclosing_statement=self.node_context.nearest_enclosing(ast.stmt),
            )
        return ret

    def _visit_comprehension_inner(self, node, typ, iterable_type):
        # need to visit the generator expression first so that we know of variables
        # created in them
        for i, generator in enumerate(node.generators):
            # for generators after the first one, compute the iterable_type inside
            # the comprehension's scope
            self.node_context.contexts.append(generator)
            try:
                self.visit_comprehension(
                    generator, iterable_type=iterable_type if i == 0 else None
                )
            finally:
                self.node_context.contexts.pop()

        with qcore.override(self, "in_comprehension_body", True):
            member_value = self.visit(node.elt)
        if member_value is UNRESOLVED_VALUE:
            return TypedValue(typ)
        else:
            return GenericValue(typ, [member_value])

    # Literals and displays

    def visit_JoinedStr(self, node):
        """JoinedStr is the node type for f-strings.

        Not too much to check here. Perhaps we can add checks that format specifiers
        are valid.

        """
        self._generic_visit_list(node.values)
        return TypedValue(str)

    def visit_Constant(self, node):
        # replaces Num, Str, etc. in 3.8+
        if isinstance(node.value, str):
            self._maybe_show_missing_f_error(node, node.value)
        return KnownValue(node.value)

    def visit_Num(self, node):
        return KnownValue(node.n)

    def visit_Str(self, node):
        self._maybe_show_implicit_non_ascii_error(node)
        self._maybe_show_missing_f_error(node, node.s)
        return KnownValue(node.s)

    def _maybe_show_missing_f_error(self, node, s):
        """Show an error if this string was probably meant to be an f-string."""
        if sys.version_info < (3, 6) or isinstance(s, bytes):
            return
        if "{" not in s:
            return
        f_str = "f" + repr(s)
        try:
            f_str_ast = ast.parse(f_str)
        except SyntaxError:
            return
        names = {
            subnode.id
            for subnode in ast.walk(f_str_ast)
            if isinstance(subnode, ast.Name)
        }
        # TODO:
        # - use nearest_enclosing() to find the Call node
        # - don't suggest this if there's (a lot of?) stuff after :
        # - some false positives with SQL queries
        # - if there are implicitly concatenated strings, the errors are correct, but
        #   we point to the wrong line and give wrong suggested fixes because the AST is weird
        if names and all(self._name_exists(name) for name in names):
            parent = self.node_context.nth_parent(2)
            if parent is not None:
                # the string is immediately .format()ed, probably doesn't need to be an f-string
                if isinstance(parent, ast.Attribute) and parent.attr == "format":
                    return
                # probably a docstring
                elif isinstance(parent, ast.Expr):
                    return
                # Probably a function that does template-like interpolation itself. In practice
                # this covers our translation API (translate("hello {user}", user=...)).
                elif isinstance(parent, ast.Call):
                    keywords = {kw.arg for kw in parent.keywords if kw.arg is not None}
                    if names <= keywords:
                        return
            self._show_error_if_checking(
                node,
                error_code=ErrorCode.missing_f,
                replacement=self.replace_node(node, f_str_ast.body[0].value),
            )

    def _name_exists(self, name):
        try:
            val = self.scope.get(name, None, VisitorState.check_names)
        except KeyError:
            return False
        else:
            return val is not UNINITIALIZED_VALUE

    def _maybe_show_implicit_non_ascii_error(self, node):
        """Strings with non-ASCII characters should be marked explicitly as bytes or unicode."""
        if six.PY3:
            return
        if not isinstance(node.s, bytes):
            return
        if not any(ord(c) > 127 for c in node.s):
            return
        if any(
            self.filename.endswith(suffix)
            for suffix in self.config.IGNORED_FILES_FOR_EXPLICIT_STRING_LITERALS
        ):
            return
        # for multiline strings, the lineno is the last line and the col_offset is -1
        # there appears to be no simple way to get to the beginning of the string, and therefore no
        # way to determine whether there is a b prefix, so just ignore these strings
        if node.col_offset == -1:
            return
        line = self._lines()[node.lineno - 1]
        char = line[node.col_offset]
        if char in ("b", "u"):
            return
        self._show_error_if_checking(
            node,
            "string containing non-ASCII characters should be explicitly marked as bytes or "
            "unicode",
            error_code=ErrorCode.implicit_non_ascii_string,
        )

    def visit_Bytes(self, node):
        return KnownValue(node.s)

    def visit_NameConstant(self, node):
        # True, False, None in py3
        return KnownValue(node.value)

    def visit_Dict(self, node):
        """Returns a KnownValue if all the keys and values can be resolved to KnownValues.

        Also checks that there are no duplicate keys and that all keys are hashable.

        """
        ret = {}
        all_pairs = []
        has_UNRESOLVED_VALUE = False
        has_star_include = False
        for key_node, value_node in zip(node.keys, node.values):
            value_val = self.visit(value_node)
            # This happens in Python 3 for syntax like "{a: b, **c}"
            if key_node is None:
                has_star_include = True
                continue
            key_val = self.visit(key_node)
            all_pairs.append((key_val, value_val))
            if not isinstance(key_val, KnownValue) or not isinstance(
                value_val, KnownValue
            ):
                has_UNRESOLVED_VALUE = True
            value = value_val.val if isinstance(value_val, KnownValue) else None

            if not isinstance(key_val, KnownValue):
                continue

            key = key_val.val

            try:
                already_exists = key in ret
            except TypeError as e:
                self._show_error_if_checking(key_node, e, ErrorCode.unhashable_key)
                continue

            if (
                already_exists
                and os.path.basename(self.filename)
                not in self.config.IGNORED_FILES_FOR_DUPLICATE_DICT_KEYS
            ):
                self._show_error_if_checking(
                    key_node,
                    "Duplicate dictionary key %r" % (key,),
                    ErrorCode.duplicate_dict_key,
                )
            ret[key] = value

        if has_star_include:
            # TODO more precise type
            return TypedValue(dict)
        elif has_UNRESOLVED_VALUE:
            return DictIncompleteValue(all_pairs)
        else:
            return KnownValue(ret)

    def visit_Set(self, node):
        return self._visit_display_read(node, set)

    def visit_List(self, node):
        return self._visit_display(node, list)

    def visit_Tuple(self, node):
        return self._visit_display(node, tuple)

    def _visit_display(self, node, typ):
        if self._is_write_ctx(node.ctx):
            if isinstance(self.being_assigned, KnownValue) and isinstance(
                self.being_assigned.val, (list, tuple)
            ):
                being_assigned = [KnownValue(val) for val in self.being_assigned.val]
            elif isinstance(self.being_assigned, SequenceIncompleteValue):
                being_assigned = self.being_assigned.members
            else:
                # TODO handle other cases; error if the object is not iterable
                being_assigned = None

            if being_assigned is not None:
                assign_to = node.elts
                if len(assign_to) != len(being_assigned):
                    # if being_assigned was empty
                    if len(being_assigned) > 0:
                        self.show_error(
                            node,
                            "Length mismatch in unpacking assignment",
                            ErrorCode.bad_unpack,
                        )
                    with qcore.override(self, "being_assigned", UNRESOLVED_VALUE):
                        self.generic_visit(node)
                else:
                    for target, value in zip(assign_to, being_assigned):
                        with qcore.override(self, "being_assigned", value):
                            self.visit(target)
            else:
                with qcore.override(self, "being_assigned", UNRESOLVED_VALUE):
                    return self.generic_visit(node)
        else:
            return self._visit_display_read(node, typ)

    def _visit_display_read(self, node, typ):
        elts = [self.visit(elt) for elt in node.elts]
        # If we have something like [*a], give up on identifying the type.
        if hasattr(ast, "Starred") and any(
            isinstance(elt, ast.Starred) for elt in node.elts
        ):
            return TypedValue(typ)
        elif all(isinstance(elt, KnownValue) for elt in elts):
            try:
                obj = typ(elt.val for elt in elts)
            except TypeError as e:
                # probably an unhashable type being included in a set
                self._show_error_if_checking(node, e, ErrorCode.unhashable_key)
                return TypedValue(typ)
            return KnownValue(obj)
        else:
            return SequenceIncompleteValue(typ, elts)

    # Operations

    def visit_BoolOp(self, node):
        val, _ = self.constraint_from_bool_op(node)
        return val

    def constraint_from_bool_op(self, node):
        """Visit an AND or OR expression.

        We want to show an error if the left operand in a BoolOp is always true,
        so we use constraint_from_condition.

        Within the BoolOp itself we set additional constraints: for an AND
        clause we know that if it is executed, all constraints to its left must
        be true, so we set a positive constraint; for OR it is the opposite, so
        we set a negative constraint.

        """
        is_and = isinstance(node.op, ast.And)
        out_constraints = []
        with self.scope.subscope():
            left = node.values[:-1]
            for condition in left:
                _, constraint = self.constraint_from_condition(condition)
                out_constraints.append(constraint)
                if is_and:
                    self.add_constraint(condition, constraint)
                else:
                    self.add_constraint(condition, constraint.invert())
            _, constraint = self._visit_possible_constraint(node.values[-1])
            out_constraints.append(constraint)
        constraint_cls = AndConstraint if is_and else OrConstraint
        constraint = reduce(constraint_cls, reversed(out_constraints))
        return UNRESOLVED_VALUE, constraint

    def visit_Compare(self, node):
        val, _ = self.constraint_from_compare(node)
        return val

    def constraint_from_compare(self, node):
        if len(node.ops) != 1:
            return self.generic_visit(node), NULL_CONSTRAINT
        op = node.ops[0]
        self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        val = UNRESOLVED_VALUE
        constraint = NULL_CONSTRAINT
        if isinstance(op, (ast.Is, ast.IsNot)):
            val = TypedValue(bool)
            if isinstance(rhs, KnownValue):
                varname = self.varname_for_constraint(node.left)
                if varname is not None:
                    positive = isinstance(op, ast.Is)
                    constraint = Constraint(
                        varname, ConstraintType.is_value, positive, rhs.val
                    )
        return val, constraint

    def visit_UnaryOp(self, node):
        val, _ = self.constraint_from_unary_op(node)
        return val

    def constraint_from_unary_op(self, node):
        if isinstance(node.op, ast.Not):
            # not doesn't have its own special method
            val, constraint = self.constraint_from_condition(node.operand)
            return val, constraint.invert()
        else:
            operand = self.visit(node.operand)
            _, method, _ = OPERATION_TO_DESCRIPTION_AND_METHOD[type(node.op)]
            val = self._check_dunder_call(node, operand, method, [], allow_call=True)
            return val, NULL_CONSTRAINT

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self._visit_binop_internal(
            node.left, left, node.op, node.right, right, node
        )

    def _visit_binop_internal(
        self, left_node, left, op, right_node, right, source_node
    ):
        # check for some py3 deprecations
        if (
            isinstance(op, ast.Div)
            and left.is_type(six.integer_types)
            and right.is_type(six.integer_types)
            and six.PY2
            and "division" not in self.future_imports
        ):
            new_node = ast.BinOp(left=left_node, op=ast.FloorDiv(), right=right_node)
            self._show_error_if_checking(
                source_node,
                error_code=ErrorCode.use_floor_div,
                replacement=self.replace_node(source_node, new_node),
            )
        elif isinstance(op, ast.Add) and (
            (left.is_type(bytes) and right.is_type(six.text_type))
            or (left.is_type(six.text_type) and right.is_type(bytes))
        ):
            self._show_error_if_checking(
                source_node, error_code=ErrorCode.mixing_bytes_and_text
            )

        # compute the return value
        # we can't use six.text_types here because we want to includes bytes in py3; six.text_types
        # is only str in py3
        if (
            isinstance(op, ast.Mod)
            and isinstance(left, KnownValue)
            and isinstance(left.val, (bytes, six.text_type))
        ):
            value, replacement_node = format_strings.check_string_format(
                left_node, left.val, right_node, right, self._show_error_if_checking
            )
            if replacement_node is not None and isinstance(source_node, ast.BinOp):
                replacement = self.replace_node(source_node, replacement_node)
                self._show_error_if_checking(
                    source_node,
                    error_code=ErrorCode.use_fstrings,
                    replacement=replacement,
                )
            return value

        # Div maps to a different dunder depending on language version and __future__ imports
        if isinstance(op, ast.Div):
            if six.PY3 or "division" in self.future_imports:
                method = "__truediv__"
                rmethod = "__rtruediv__"
            else:
                method = "__div__"
                rmethod = "__rdiv__"
            description = "division"
        else:
            description, method, rmethod = OPERATION_TO_DESCRIPTION_AND_METHOD[type(op)]
        allow_call = method not in self.config.DISALLOW_CALLS_TO_DUNDERS
        with self.catch_errors() as left_errors:
            left_result = self._check_dunder_call(
                source_node, left, method, [right], allow_call=allow_call,
            )
        if not left_errors:
            return left_result

        with self.catch_errors() as right_errors:
            right_result = self._check_dunder_call(
                source_node, right, rmethod, [left], allow_call=allow_call,
            )
        if not right_errors:
            return right_result

        self.show_error(
            source_node,
            "Unsupported operands for %s: %s and %s" % (description, left, right),
            error_code=ErrorCode.unsupported_operation,
        )
        return UNRESOLVED_VALUE

    # Indexing

    def visit_Ellipsis(self, node):
        return KnownValue(Ellipsis)

    def visit_Slice(self, node):
        lower = self.visit(node.lower) if node.lower is not None else None
        upper = self.visit(node.upper) if node.upper is not None else None
        step = self.visit(node.step) if node.step is not None else None

        if all(isinstance(val, KnownValue) for val in (lower, upper, step)):
            return KnownValue(slice(lower, upper, step))
        else:
            return TypedValue(slice)

    def visit_ExtSlice(self, node):
        dims = [self.visit(dim) for dim in node.dims]
        if all(isinstance(val, KnownValue) for val in dims):
            return KnownValue(tuple(val.val for val in dims))
        else:
            return SequenceIncompleteValue(tuple, dims)

    def visit_Index(self, node):
        return self.visit(node.value)

    # Control flow

    def visit_Await(self, node):
        value = self.visit(node.value)
        if isinstance(value, AwaitableIncompleteValue):
            return value.value
        else:
            return self._check_dunder_call(node.value, value, "__await__", [])

    def visit_YieldFrom(self, node):
        self.is_generator = True
        value = self.visit(node.value)
        if not TypedValue(Iterable).is_value_compatible(value):
            self._show_error_if_checking(
                node,
                "Cannot use %s in yield from" % (value,),
                error_code=ErrorCode.bad_yield_from,
            )
        return UNRESOLVED_VALUE

    def visit_Yield(self, node):
        if self._is_checking():
            if self.in_comprehension_body:
                self._show_error_if_checking(
                    node, error_code=ErrorCode.yield_in_comprehension
                )

            ctx = self.yield_checker.check_yield(node, self.current_statement)
        else:
            ctx = qcore.empty_context
        with ctx:
            if node.value is not None:
                value = self.visit(node.value)
            else:
                value = None

        if node.value is None and self.async_kind in (
            AsyncFunctionKind.normal,
            AsyncFunctionKind.pure,
        ):
            self._show_error_if_checking(node, error_code=ErrorCode.yield_without_value)
        self.is_generator = True

        # unwrap the results of async yields
        if self.async_kind != AsyncFunctionKind.non_async and value is not None:
            return self._unwrap_yield_result(node, value)
        else:
            return UNRESOLVED_VALUE

    def _unwrap_yield_result(self, node, value):
        if isinstance(value, AsyncTaskIncompleteValue):
            return value.value
        elif isinstance(value, TypedValue) and issubclass(value.typ, (list, tuple)):
            if isinstance(value, SequenceIncompleteValue):
                values = [
                    self._unwrap_yield_result(node, member) for member in value.members
                ]
                if all(isinstance(member, KnownValue) for member in values):
                    return KnownValue(value.typ(member.val for member in values))
                else:
                    return SequenceIncompleteValue(value.typ, values)
            elif isinstance(value, GenericValue):
                member_value = self._unwrap_yield_result(node, value.get_arg(0))
                return GenericValue(value.typ, [member_value])
            else:
                return value
        elif isinstance(value, TypedValue) and issubclass(value.typ, dict):
            if isinstance(value, DictIncompleteValue):
                values = [
                    (key, self._unwrap_yield_result(node, val))
                    for key, val in value.items
                ]
                return DictIncompleteValue(values)
            elif isinstance(value, GenericValue):
                val = self._unwrap_yield_result(node, value.get_arg(1))
                return GenericValue(value.typ, [value.get_arg(0), val])
            else:
                return value
        elif isinstance(value, KnownValue) and isinstance(value.val, asynq.ConstFuture):
            return KnownValue(value.val.value())
        elif isinstance(value, KnownValue) and value.val is None:
            return value  # we're allowed to yield None
        elif isinstance(value, KnownValue) and isinstance(value.val, (list, tuple)):
            values = [
                self._unwrap_yield_result(node, KnownValue(elt)) for elt in value.val
            ]
            return KnownValue(values)
        elif value is UNRESOLVED_VALUE:
            return UNRESOLVED_VALUE
        elif isinstance(value, MultiValuedValue):
            return unite_values(
                *[self._unwrap_yield_result(node, val) for val in value.vals]
            )
        elif value.is_type((asynq.FutureBase, asynq.AsyncTask)):
            return UNRESOLVED_VALUE
        else:
            self._show_error_if_checking(
                node,
                "Invalid value yielded: %r" % (value,),
                error_code=ErrorCode.bad_async_yield,
            )
            return UNRESOLVED_VALUE

    def visit_Return(self, node):
        """For return type inference, set the pseudo-variable RETURN_VALUE in the local scope."""
        if node.value is None:
            value = KnownValue(None)
            method_return_type.check_no_return(node, self, self.current_function_name)
        else:
            value = self.visit(node.value)
            method_return_type.check_return_value(
                node, self, value, self.current_function_name
            )
        self.return_values.append(value)
        self._set_name_in_scope(LEAVES_SCOPE, node, UNRESOLVED_VALUE)
        if (
            # TODO check generator types properly
            not (self.is_generator and self.async_kind == AsyncFunctionKind.non_async)
            and self.expected_return_value is not None
            and not self.is_value_compatible(self.expected_return_value, value)
        ):
            self._show_error_if_checking(
                node,
                "Declared return type %s is incompatible with actual return type %s"
                % (self.expected_return_value, value),
                error_code=ErrorCode.incompatible_return_value,
            )
        elif self.expected_return_value == KnownValue(None) and value != KnownValue(
            None
        ):
            self._show_error_if_checking(
                node,
                "Function declared as returning None may not return a value",
                error_code=ErrorCode.incompatible_return_value,
            )

    def visit_Raise(self, node):
        # we need to record this in the return value so that functions that always raise
        # NotImplementedError aren't inferred as returning None
        self.return_values.append(None)
        self._set_name_in_scope(LEAVES_SCOPE, node, UNRESOLVED_VALUE)

        raised_expr = node.type if six.PY2 else node.exc

        if raised_expr is not None:
            raised_value = self.visit(raised_expr)
            if isinstance(raised_value, KnownValue):
                val = raised_value.val
                # technically you can also raise an instance of an old-style class but we don't care
                if not (
                    isinstance(val, BaseException)
                    or (inspect.isclass(val) and issubclass(val, BaseException))
                ):
                    self._show_error_if_checking(
                        node, error_code=ErrorCode.bad_exception
                    )
            elif isinstance(raised_value, TypedValue):
                typ = raised_value.typ
                # we let you do except Exception as e: raise type(e)
                if not (issubclass(typ, BaseException) or typ is type):
                    self._show_error_if_checking(
                        node, error_code=ErrorCode.bad_exception
                    )
            # TODO handle other values

        if six.PY2:
            # these two are deprecated, maybe we should error on them
            if node.inst is not None:
                self.visit(node.inst)
            if node.tback is not None:
                self.visit(node.tback)
        else:
            if node.cause is not None:
                self.visit(node.cause)

    def visit_Assert(self, node):
        test, constraint = self._visit_possible_constraint(node.test)
        if node.msg is not None:
            with self.scope.subscope():
                self.add_constraint(node, constraint.invert())
                self.visit(node.msg)
        self.add_constraint(node, constraint)
        # code after an assert False is unreachable
        val = boolean_value(test)
        if val is True:
            self._show_error_if_checking(
                node, error_code=ErrorCode.condition_always_true
            )
        elif val is False:
            self._set_name_in_scope(LEAVES_SCOPE, node, UNRESOLVED_VALUE)

    def add_constraint(self, node, constraint):
        self.scope.current_scope().add_constraint(constraint, node, self.state)

    def _visit_possible_constraint(self, node):
        if isinstance(node, ast.Compare):
            return self.constraint_from_compare(node)
        elif isinstance(node, ast.Name):
            constraint = Constraint(node.id, ConstraintType.is_truthy, True, None)
            return self.visit(node), constraint
        elif isinstance(node, ast.Attribute):
            varname = self.varname_for_constraint(node)
            val = self.visit(node)
            if varname is not None:
                constraint = Constraint(varname, ConstraintType.is_truthy, True, None)
                return val, constraint
            else:
                return val, NULL_CONSTRAINT
        elif isinstance(node, ast.Call):
            return self.constraint_from_call(node)
        elif isinstance(node, ast.UnaryOp):
            return self.constraint_from_unary_op(node)
        elif isinstance(node, ast.BoolOp):
            return self.constraint_from_bool_op(node)
        else:
            return self.visit(node), NULL_CONSTRAINT

    def visit_Break(self, node):
        self._set_name_in_scope(LEAVES_LOOP, node, UNRESOLVED_VALUE)

    def visit_Continue(self, node):
        self._set_name_in_scope(LEAVES_LOOP, node, UNRESOLVED_VALUE)

    def visit_For(self, node):
        iterated_value, num_members = self._member_value_of_iterator(node.iter)
        if self.config.FOR_LOOP_ALWAYS_ENTERED:
            always_entered = True
        else:
            always_entered = num_members is not None and num_members > 0
        with self.scope.subscope() as body_scope:
            with self.scope.loop_scope():
                with qcore.override(self, "being_assigned", iterated_value):
                    # assume that node.target is not affected by variable assignments in the body
                    # one could write some contortion like
                    # for (a if a[0] == 1 else b)[0] in range(2):
                    #   b = [1]
                    # but that doesn't seem worth supporting
                    self.visit(node.target)
                self._generic_visit_list(node.body)
        self._handle_loop_else(node.orelse, body_scope, always_entered)

        # in loops, variables may have their first read before their first write
        # see e.g. test_stacked_scopes.TestLoop.test_conditional_in_loop
        # to get all the definition nodes in that case, visit the body twice in the collecting
        # phase
        if self.state == VisitorState.collect_names:
            with self.scope.subscope():
                with qcore.override(self, "being_assigned", iterated_value):
                    self.visit(node.target)
                self._generic_visit_list(node.body)

    def visit_While(self, node):
        # see comments under For for discussion
        test, constraint = self.constraint_from_condition(node.test)
        always_entered = boolean_value(test) is True
        with self.scope.subscope() as body_scope:
            with self.scope.loop_scope():
                # The "node" argument need not be an AST node but must be unique.
                self.add_constraint((node, 1), constraint)
                self._generic_visit_list(node.body)
        self._handle_loop_else(node.orelse, body_scope, always_entered)

        if self.state == VisitorState.collect_names:
            self.visit(node.test)
            with self.scope.subscope():
                self.add_constraint((node, 2), constraint)
                self._generic_visit_list(node.body)

    def _handle_loop_else(self, orelse, body_scope, always_entered):
        if always_entered:
            self.scope.combine_subscopes([body_scope])
            # Replace body_scope with a dummy scope, because body_scope
            # should always execute and has already been combined in.
            with self.scope.subscope() as body_scope:
                pass
        with self.scope.subscope() as else_scope:
            self._generic_visit_list(orelse)
        self.scope.combine_subscopes([body_scope, else_scope])

    def _member_value_of_iterator(self, node, is_async=False):
        """Analyze an iterator AST node.

        Returns a tuple of two values:
        - A Value object representing a member of the iterator.
        - The number of elements in the iterator, or None if the number is unknown.

        """
        iterated = self.visit(node)
        if is_async:
            return self._member_value_of_async_iterator_val(iterated, node)
        return self._member_value_of_iterator_val(iterated, node)

    def _member_value_of_async_iterator_val(self, iterated, node):
        iterator = self._check_dunder_call(node, iterated, "__aiter__", [])
        return self._check_dunder_call(node, iterator, "__anext__", []), None

    def _member_value_of_iterator_val(self, iterated, node):
        if isinstance(iterated, KnownValue):
            if iterated.val is not None and not analysis_lib.is_iterable(iterated.val):
                self._show_error_if_checking(
                    node,
                    "Object %r is not iterable" % (iterated.val,),
                    ErrorCode.unsupported_operation,
                )
                return UNRESOLVED_VALUE, None
            if iterated.is_type(range):
                return TypedValue(int), len(iterated.val)
            # if the thing we're iterating over is e.g. a file or an infinite generator, calling
            # list() may hang the process
            if not iterated.is_type((list, set, tuple, dict, six.string_types)):
                return UNRESOLVED_VALUE, None
            try:
                values = list(iterated.val)
            except Exception:
                # we couldn't iterate over it for whatever reason; just ignore for now
                return UNRESOLVED_VALUE, None
            return unite_values(*map(KnownValue, values)), len(values)
        elif isinstance(iterated, SequenceIncompleteValue):
            return unite_values(*iterated.members), len(iterated.members)
        elif isinstance(iterated, DictIncompleteValue):
            return (
                unite_values(*[key for key, _ in iterated.items]),
                len(iterated.items),
            )
        elif isinstance(iterated, TypedValue):
            if not issubclass(
                iterated.typ, collections.Iterable
            ) and not self._should_ignore_type(iterated.typ):
                self._show_error_if_checking(
                    node,
                    "Object of type %r is not iterable" % (iterated.typ,),
                    ErrorCode.unsupported_operation,
                )
            if isinstance(iterated, GenericValue):
                return iterated.get_arg(0), None
        elif isinstance(iterated, MultiValuedValue):
            vals, nums = zip(
                *[
                    self._member_value_of_iterator_val(val, node)
                    for val in iterated.vals
                ]
            )
            num = nums[0] if len(set(nums)) == 1 else None
            return unite_values(*vals), num
        return UNRESOLVED_VALUE, None

    def visit_TryExcept(self, node):
        # reset yield checks between branches to avoid incorrect errors when we yield both in the
        # try and the except block
        # this node type is py2 only (py3 uses Try), but the visit_Try implementation delegates to
        # this method
        with self.scope.subscope():
            with self.scope.subscope() as try_scope:
                self._generic_visit_list(node.body)
                self.yield_checker.reset_yield_checks()
                self._generic_visit_list(node.orelse)
            with self.scope.subscope() as dummy_subscope:
                pass
            self.scope.combine_subscopes([try_scope, dummy_subscope])

            except_scopes = []
            for handler in node.handlers:
                with self.scope.subscope() as except_scope:
                    except_scopes.append(except_scope)
                    self.yield_checker.reset_yield_checks()
                    self.visit(handler)

        self.scope.combine_subscopes([try_scope] + except_scopes)
        self.yield_checker.reset_yield_checks()

    def visit_TryFinally(self, node):
        # We visit the finally block twice, representing the two possible code paths where the try
        # body does and does not raise an exception.
        with self.scope.subscope() as try_scope:
            self._generic_visit_list(node.body)
            self._generic_visit_list(node.finalbody)
        with self.scope.subscope():
            self._generic_visit_list(node.finalbody)
        self.scope.combine_subscopes([try_scope])

    def visit_Try(self, node):
        # py3 combines the Try and Try/Finally nodes
        if node.finalbody:
            with self.scope.subscope() as try_scope:
                self.visit_TryExcept(node)
                self._generic_visit_list(node.finalbody)
            with self.scope.subscope():
                self._generic_visit_list(node.finalbody)
            self.scope.combine_subscopes([try_scope])
        else:
            self.visit_TryExcept(node)

    def visit_ExceptHandler(self, node):
        if node.type is not None:
            typ = self.visit(node.type)
            if isinstance(typ, KnownValue):
                val = typ.val
                if isinstance(val, tuple):
                    if all(self._check_valid_exception_class(cls, node) for cls in val):
                        to_assign = unite_values(*[TypedValue(cls) for cls in val])
                    else:
                        to_assign = TypedValue(BaseException)
                else:
                    if self._check_valid_exception_class(val, node):
                        to_assign = TypedValue(val)
                    else:
                        to_assign = TypedValue(BaseException)
            else:
                # maybe this should be an error, exception classes should virtually always be
                # statically findable
                to_assign = TypedValue(BaseException)
            if node.name is not None:
                if isinstance(node.name, str):  # py3
                    self._set_name_in_scope(node.name, node, value=to_assign)
                else:
                    with qcore.override(self, "being_assigned", to_assign):
                        self.visit(node.name)

        self._generic_visit_list(node.body)

    def _check_valid_exception_class(self, val, node):
        if not (isinstance(val, type) and issubclass(val, BaseException)):
            self._show_error_if_checking(
                node,
                "%r is not an exception class" % (val,),
                error_code=ErrorCode.bad_except_handler,
            )
            return False
        else:
            return True

    def visit_If(self, node):
        _, constraint = self.constraint_from_condition(node.test)
        # reset yield checks to avoid incorrect errors when we yield in both the condition and one
        # of the blocks
        self.yield_checker.reset_yield_checks()
        with self.scope.subscope() as body_scope:
            self.add_constraint(node, constraint)
            self._generic_visit_list(node.body)
        self.yield_checker.reset_yield_checks()

        with self.scope.subscope() as else_scope:
            self.add_constraint(node, constraint.invert())
            self._generic_visit_list(node.orelse)
        self.scope.combine_subscopes([body_scope, else_scope])
        self.yield_checker.reset_yield_checks()

    def visit_IfExp(self, node):
        _, constraint = self.constraint_from_condition(node.test)
        with self.scope.subscope():
            self.add_constraint(node, constraint)
            then_val = self.visit(node.body)
        with self.scope.subscope():
            self.add_constraint(node, constraint.invert())
            else_val = self.visit(node.orelse)
        return unite_values(then_val, else_val)

    def constraint_from_condition(self, node):
        condition, constraint = self._visit_possible_constraint(node)
        if self._is_collecting():
            return condition, constraint
        if condition is not None:
            typ = condition.get_type()
            if typ is not None and not self._can_be_used_as_boolean(typ):
                self._show_error_if_checking(
                    node,
                    "Object of type {} will always evaluate to True in boolean context".format(
                        typ
                    ),
                    error_code=ErrorCode.non_boolean_in_boolean_context,
                )
        return TypedValue(bool), constraint

    def _can_be_used_as_boolean(self, typ):
        if hasattr(typ, "__len__"):
            return True
        if hasattr(typ, _BOOL_DUNDER):
            # asynq futures have __bool__, but it always throws an error to help find bugs at
            # runtime, so special-case it. If necessary we could make this into a configuration
            # option.
            bool_fn = getattr(typ, _BOOL_DUNDER)
            return bool_fn is not getattr(asynq.FutureBase, _BOOL_DUNDER)
        return False

    def visit_Expr(self, node):
        value = self.visit(node.value)
        if value.is_type((asynq.FutureBase, asynq.AsyncTask)):
            new_node = ast.Expr(value=ast.Yield(value=node.value))
            replacement = self.replace_node(node, new_node)
            self._show_error_if_checking(
                node, error_code=ErrorCode.task_needs_yield, replacement=replacement
            )
        elif isinstance(value, AwaitableIncompleteValue):
            if self.is_async_def:
                new_node = ast.Expr(value=ast.Await(value=node.value))
            else:
                new_node = ast.Expr(value=ast.YieldFrom(value=node.value))
            replacement = self.replace_node(node, new_node)
            self._show_error_if_checking(
                node, error_code=ErrorCode.missing_await, replacement=replacement
            )
        if value is NO_RETURN_VALUE:
            self._set_name_in_scope(LEAVES_SCOPE, node, UNRESOLVED_VALUE)
        return value

    # Assignments

    def visit_Assign(self, node):
        is_yield = isinstance(node.value, ast.Yield)
        value = self.visit(node.value)

        with qcore.override(
            self, "being_assigned", value
        ), self.yield_checker.check_yield_result_assignment(is_yield):
            # syntax like 'x = y = 0' results in multiple targets
            self._generic_visit_list(node.targets)

    def visit_AnnAssign(self, node):
        annotation = self.visit(node.annotation)
        if isinstance(annotation, KnownValue) and is_typing_name(
            annotation.val, "Final"
        ):
            # TODO disallow assignments to Final variables (current code
            # just avoids false positive errors).
            is_final = True
            expected_type = UNRESOLVED_VALUE
        else:
            expected_type = self._value_of_annotation_type(annotation, node.annotation)
            is_final = False

        if node.value:
            is_yield = isinstance(node.value, ast.Yield)
            value = self.visit(node.value)
            if not self.is_value_compatible(expected_type, value):
                self._show_error_if_checking(
                    node.value,
                    "Incompatible assignment: expected %s, got %s"
                    % (expected_type, value),
                    error_code=ErrorCode.incompatible_assignment,
                )

            with qcore.override(
                self, "being_assigned", value if is_final else expected_type
            ), self.yield_checker.check_yield_result_assignment(is_yield):
                self.visit(node.target)
        # TODO: Idea for what to do if there is no value:
        # - Scopes keep track of a map {name: expected type}
        # - Assignments that are inconsistent with the declared type produce an error.

    def visit_AugAssign(self, node):
        is_yield = isinstance(node.value, ast.Yield)
        rhs = self.visit(node.value)

        if isinstance(node.target, ast.Name):
            lhs = self.visit_Name(node.target, force_read=True)
        else:
            lhs = UNRESOLVED_VALUE

        value = self._visit_binop_internal(
            node.target, lhs, node.op, node.value, rhs, node
        )

        with qcore.override(
            self, "being_assigned", value
        ), self.yield_checker.check_yield_result_assignment(is_yield):
            # syntax like 'x = y = 0' results in multiple targets
            self.visit(node.target)

    def visit_Name(self, node, force_read=False):
        if force_read or self._is_read_ctx(node.ctx):
            self.yield_checker.record_usage(node.id, node)
            value = self.resolve_name(node)
            if isinstance(value, KnownValue):
                self._maybe_record_usage(value.val)
            varname_value = VariableNameValue.from_varname(
                node.id, self.config.varname_value_map()
            )
            if varname_value is not None and self._should_use_varname_value(value):
                value = varname_value
            value = self._maybe_use_hardcoded_type(value, node.id)
            return value
        elif self._is_write_ctx(node.ctx):
            self.yield_checker.record_assignment(node.id)
            self._set_name_in_scope(node.id, node, value=self.being_assigned)
            if self._name_node_to_statement is not None:
                statement = self.node_context.nearest_enclosing(
                    (ast.stmt, ast.comprehension)
                )
                self._name_node_to_statement[node] = statement
            return None
        else:
            # not sure when (if ever) the other contexts can happen
            self.show_error(
                node, "Bad context: %s" % (node.ctx,), ErrorCode.unexpected_node
            )
            return None

    def visit_arg(self, node):
        # py3 only; in py2 arguments are just Name nodes
        self.yield_checker.record_assignment(node.arg)
        self._set_name_in_scope(node.arg, node, value=self.being_assigned)

    def _should_use_varname_value(self, value):
        """Returns whether a value should be replaced with VariableNameValue.

        VariableNameValues are used for things like uids that are represented as integers, so we
        mostly only want to replace integers. More generally, this function exists to avoid
        throwing away more useful information like a KnownValue. Therefore, we want to avoid
        replacing a function with a name that happens to ends in _uid, because knowing the exact
        function being called gives more information than having a VariableNameValue. On the other
        hand, an UNRESOLVED_VALUE carries no information, so we're fine always replacing it.

        """
        if isinstance(value, KnownValue):
            return type(value.val) in six.integer_types
        elif (
            type(value) is TypedValue
        ):  # Only replace exactly TypedValue(int), not subtypes
            return value.typ in six.integer_types
        else:
            return value is UNRESOLVED_VALUE

    def _maybe_use_hardcoded_type(self, value, name):
        """Replaces a value with a name of hardcoded type where applicable."""
        if value is not UNRESOLVED_VALUE and not isinstance(value, MultiValuedValue):
            return value

        try:
            typ = self.config.NAMES_OF_KNOWN_TYPE[name]
        except KeyError:
            return value
        else:
            return TypedValue(typ)

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        index = self.visit(node.slice)

        if value.is_type((list, tuple) + six.string_types) and not index.is_type(slice):
            index = self._check_dunder_call(
                node, index, "__index__", [], allow_call=True
            )

        if isinstance(node.ctx, ast.Store):
            return self._check_dunder_call(
                node.value, value, "__setitem__", [index, self.being_assigned]
            )
        elif isinstance(node.ctx, ast.Load):
            if isinstance(value, TypedDictValue):
                if not TypedValue(str).is_value_compatible(index):
                    self._show_error_if_checking(
                        node.slice.value,
                        "dict key must be str, not %s" % (index,),
                        error_code=ErrorCode.invalid_typeddict_key,
                    )
                    return UNRESOLVED_VALUE
                elif isinstance(index, KnownValue):
                    try:
                        return value.items[index.val]
                    # probably KeyError, but catch anything in case it's an
                    # unhashable str subclass or something
                    except Exception:
                        # No error here; TypedDicts may have additional keys at runtime.
                        pass
            if isinstance(value, SequenceIncompleteValue):
                if isinstance(index, KnownValue):
                    if isinstance(index.val, int) and -len(
                        value.members
                    ) <= index.val < len(value.members):
                        return value.members[index.val]
                    # Don't error if it's out of range; the object may be mutated at runtime.
                    # TODO: handle slices; error for things that aren't ints or slices.

            with self.catch_errors() as getitem_errors:
                return_value = self._check_dunder_call(
                    node.value, value, "__getitem__", [index], allow_call=True
                )
            if getitem_errors:

                def on_error(typ):
                    self._show_error_if_checking(
                        node,
                        "Object %s does not support subscripting" % (value,),
                        error_code=ErrorCode.unsupported_operation,
                    )
                    return UNRESOLVED_VALUE

                with self.catch_errors() as class_getitem_errors:
                    cgi = self._get_attribute(
                        node.value, "__class_getitem__", value, on_error=on_error
                    )
                    return_value, _ = self._get_argspec_and_check_call(
                        node.value, cgi, [index], allow_call=True
                    )
                if class_getitem_errors:
                    # if __class_getitem__ didn't work either, show __getitem__ errors
                    self.show_caught_errors(getitem_errors)
            if (
                return_value is UNRESOLVED_VALUE
                and isinstance(index, KnownValue)
                and isinstance(index.val, six.string_types)
            ):
                varname_value = VariableNameValue.from_varname(
                    index.val, self.config.varname_value_map()
                )
                if varname_value is not None:
                    return_value = varname_value
            return return_value
        elif isinstance(node.ctx, ast.Del):
            return self._check_dunder_call(node.value, value, "__delitem__", [index])
        else:
            self.show_error(
                node,
                "Unexpected subscript context: %s" % (node.ctx,),
                ErrorCode.unexpected_node,
            )
            return UNRESOLVED_VALUE

    def _check_dunder_call(self, node, callee_val, method_name, args, allow_call=False):
        lookup_val = callee_val.get_type_value()

        def on_error(typ):
            self._show_error_if_checking(
                node,
                "Object of type %s does not support %r" % (callee_val, method_name),
                error_code=ErrorCode.unsupported_operation,
            )
            return UNRESOLVED_VALUE

        method_object = self._get_attribute(
            node, method_name, lookup_val, on_error=on_error
        )
        return_value, _ = self._get_argspec_and_check_call(
            node, method_object, [callee_val] + args, allow_call=allow_call
        )
        return return_value

    def visit_Attribute(self, node):
        """Visits an Attribute node (e.g. a.b).

        This resolves the value on the left and checks that it has the attribute. If it does not, an
        error is shown, unless the node matches IGNORED_PATHS or IGNORED_END_OF_REFERENCE.

        """

        if isinstance(node.value, ast.Name):
            attr_str = "%s.%s" % (node.value.id, node.attr)
            if self._is_write_ctx(node.ctx):
                self.yield_checker.record_assignment(attr_str)
            else:
                self.yield_checker.record_usage(attr_str, node)

        root_value = self.visit(node.value)
        if self._is_write_ctx(node.ctx):
            return self._visit_set_attribute(node, root_value)
        elif self._is_read_ctx(node.ctx):
            if self._is_checking():
                self.asynq_checker.record_attribute_access(root_value, node.attr, node)
            value = self._get_attribute(node, node.attr, root_value)
            if self._should_use_varname_value(value):
                varname_value = VariableNameValue.from_varname(
                    node.attr, self.config.varname_value_map()
                )
                if varname_value is not None:
                    return varname_value
            if self.scope.scope_type() == ScopeType.function_scope:
                composite = self.varname_for_constraint(node)
                if composite:
                    local_value = self.scope.current_scope().get_local(
                        composite, node, self.state, fallback_value=value
                    )
                    if isinstance(local_value, MultiValuedValue):
                        vals = [
                            val
                            for val in local_value.vals
                            if val is not UNINITIALIZED_VALUE
                        ]
                        if vals:
                            local_value = unite_values(*vals)
                        else:
                            local_value = UNINITIALIZED_VALUE
                    if local_value is not UNINITIALIZED_VALUE:
                        value = local_value
            value = self._maybe_use_hardcoded_type(value, node.attr)
            return value
        else:
            self.show_error(node, "Unknown context", ErrorCode.unexpected_node)
            return None

    def _get_attribute(self, node, attr, root_value, on_error=None):
        if on_error is None:

            def on_error(typ):
                # default on_error() doesn't throw an error in many
                # cases where we're not quite suer whether an attribute
                # will exist.
                if isinstance(root_value, UnboundMethodValue):
                    if self._should_ignore_val(node):
                        return UNRESOLVED_VALUE
                elif isinstance(root_value, KnownValue):
                    # super calls on mixin classes may use attributes that are defined only on child classes
                    if isinstance(typ, super):
                        subclasses = qcore.inspection.get_subclass_tree(
                            typ.__thisclass__
                        )
                        if any(
                            hasattr(cls, attr)
                            for cls in subclasses
                            if cls is not typ.__thisclass__
                        ):
                            return UNRESOLVED_VALUE

                    # Ignore objects that override __getattr__
                    if (
                        _static_hasattr(typ, "__getattr__")
                        or self._should_ignore_val(node)
                        or _safe_getattr(typ, "_pyanalyze_is_nested_function", False)
                    ):
                        return UNRESOLVED_VALUE
                else:
                    # namedtuples have only static attributes
                    if not (
                        isinstance(typ, type)
                        and issubclass(typ, tuple)
                        and not hasattr(typ, "__getattr__")
                    ):
                        return self._maybe_get_attr_value(typ, attr)
                self._show_error_if_checking(
                    node,
                    "%s has no attribute %r" % (root_value, attr),
                    ErrorCode.undefined_attribute,
                )
                return UNRESOLVED_VALUE

        if isinstance(root_value, KnownValue):
            return self._visit_get_attribute_known(node, attr, root_value.val, on_error)
        elif isinstance(root_value, SubclassValue):
            return self._visit_get_attribute_subclass(
                node, attr, root_value.typ, on_error
            )
        elif isinstance(root_value, TypedValue):
            return self._visit_get_attribute_typed(node, attr, root_value.typ, on_error)
        elif isinstance(root_value, UnboundMethodValue):
            return self._visit_get_attribute_unbound(node, attr, root_value, on_error)
        else:
            return UNRESOLVED_VALUE

    def _visit_get_attribute_subclass(self, node, attr, typ, on_error):
        self._record_type_attr_read(typ, attr, node)

        # First check values that are special in Python
        if attr == "__class__":
            return KnownValue(type(typ))
        elif attr == "__dict__":
            return TypedValue(dict)
        elif attr == "__bases__":
            return GenericValue(tuple, [SubclassValue(object)])
        return self._get_attribute_from_mro(
            node, attr, typ, self._get_attribute_value_from_raw_subclass, on_error
        )

    def _get_attribute_value_from_raw_subclass(self, cls_val, attr, typ, node):
        if (
            qcore.inspection.is_classmethod(cls_val)
            or inspect.ismethod(cls_val)
            or inspect.isfunction(cls_val)
            or isinstance(cls_val, (MethodDescriptorType, SlotWrapperType))
            or (
                # non-static method
                _static_hasattr(cls_val, "decorator")
                and _static_hasattr(cls_val, "instance")
                and not isinstance(cls_val.instance, type)
            )
            or asynq.is_async_fn(cls_val)
        ):
            # static or class method
            return KnownValue(cls_val)
        elif _static_hasattr(cls_val, "__get__"):
            return UNRESOLVED_VALUE  # can't figure out what this will return
        elif self.config.should_ignore_class_attribute(cls_val):
            return UNRESOLVED_VALUE  # probably set on child classes
        else:
            return self._maybe_replace_name_attribute(node, attr, typ, cls_val)

    def _visit_get_attribute_typed(self, node, attr, typ, on_error):
        self._record_type_attr_read(typ, attr, node)

        # First check values that are special in Python
        if attr == "__class__":
            return KnownValue(typ)
        elif attr == "__dict__":
            return TypedValue(dict)
        return self._get_attribute_from_mro(
            node, attr, typ, self._get_attribute_value_from_raw_typed, on_error
        )

    def _get_attribute_from_mro(self, node, attr, typ, attribute_from_raw, on_error):
        # Then go through the MRO and find base classes that may define the attribute.
        try:
            mro = list(typ.mro())
        except Exception:
            # broken mro method
            pass
        else:
            for base_cls in mro:
                try:
                    # Make sure to use only __annotations__ that are actually on this
                    # class, not ones inherited from a base class.
                    annotation = base_cls.__dict__["__annotations__"][attr]
                except Exception:
                    # no __annotations__, or it's not a dict, or the attr isn't there
                    try:
                        # Make sure we use only the object from this class, but do invoke
                        # the descriptor protocol with getattr.
                        base_cls.__dict__[attr]
                        cls_val = getattr(typ, attr)
                    except Exception:
                        pass
                    else:
                        return attribute_from_raw(cls_val, attr, typ, node)
                else:
                    return type_from_runtime(annotation)

        attrs_type = _get_attrs_attribute(typ, attr)
        if attrs_type is not None:
            return attrs_type

        # Even if we didn't find it any __dict__, maybe getattr() finds it directly.
        try:
            cls_val = getattr(typ, attr)
        except Exception:
            pass
        else:
            return attribute_from_raw(cls_val, attr, typ, node)

        return on_error(typ)

    def _get_attribute_value_from_raw_typed(self, cls_val, attr, typ, node):
        if isinstance(cls_val, property):
            if cls_val in self.config.PROPERTIES_OF_KNOWN_TYPE:
                return self.config.PROPERTIES_OF_KNOWN_TYPE[cls_val]
            argspec = self.arg_spec_cache.get_argspec(cls_val)
            if argspec is not None:
                if argspec.has_return_value():
                    return argspec.return_value
                # If we visited the property and inferred a return value,
                # use it.
                if id(argspec) in self._argspec_to_retval:
                    return self._argspec_to_retval[id(argspec)]
            return UNRESOLVED_VALUE
        elif qcore.inspection.is_classmethod(cls_val):
            return KnownValue(cls_val)
        elif inspect.ismethod(cls_val):
            return UnboundMethodValue(node.attr, typ)
        elif inspect.isfunction(cls_val):
            if six.PY3:
                # either a staticmethod or an unbound method
                try:
                    descriptor = inspect.getattr_static(typ, node.attr)
                except AttributeError:
                    # probably a super call; assume unbound method
                    if attr != "__new__":
                        return UnboundMethodValue(attr, typ)
                    else:
                        # __new__ is implicitly a staticmethod
                        return KnownValue(cls_val)
                if isinstance(descriptor, staticmethod) or node.attr == "__new__":
                    return KnownValue(cls_val)
                else:
                    return UnboundMethodValue(node.attr, typ)
            else:
                # staticmethod
                return KnownValue(cls_val)
        elif isinstance(cls_val, (MethodDescriptorType, SlotWrapperType)):
            # built-in method; e.g. scope_lib.tests.SimpleDatabox.get
            return UnboundMethodValue(attr, typ)
        elif (
            _static_hasattr(cls_val, "decorator")
            and _static_hasattr(cls_val, "instance")
            and not isinstance(cls_val.instance, type)
        ):
            # non-static method
            return UnboundMethodValue(node.attr, typ)
        elif asynq.is_async_fn(cls_val):
            # static or class method
            return KnownValue(cls_val)
        elif _static_hasattr(cls_val, "__get__"):
            try:
                is_known = cls_val in self.config.PROPERTIES_OF_KNOWN_TYPE
            except TypeError:  # not hashable
                is_known = False
            if is_known:
                return self.config.PROPERTIES_OF_KNOWN_TYPE[cls_val]
            else:
                return UNRESOLVED_VALUE  # can't figure out what this will return
        elif self.config.should_ignore_class_attribute(cls_val):
            return UNRESOLVED_VALUE  # probably set on child classes
        else:
            return self._maybe_replace_name_attribute(node, attr, typ, cls_val)

    def _visit_get_attribute_known(self, node, attr, value, on_error):
        self._record_type_attr_read(type(value), attr, node)

        if value is None:
            # This usually indicates some context is set to None
            # in the module and initialized later.
            return UNRESOLVED_VALUE

        def _get_attribute_value_from_raw_known(cls_val, attr, typ, node):
            self._maybe_record_usage(cls_val)
            return self._maybe_replace_name_attribute(node, attr, value, cls_val)

        return self._get_attribute_from_mro(
            node, attr, value, _get_attribute_value_from_raw_known, on_error
        )

    def _maybe_replace_name_attribute(self, node, attr, value, attr_value):
        # the __name__ of a class is bytes in py2 but text in py3, but treating it as bytes, can
        # often produce unnecessary errors about mixing bytes and text in py2; such mixing is
        # safe because __name__ in py2 is always ascii anyway
        if (
            attr in ("__name__", "__module__")
            and isinstance(value, type)
            and isinstance(attr_value, bytes)
        ):
            return unite_values(
                KnownValue(attr_value, source_node=node),
                KnownValue(attr_value.decode("ascii"), source_node=node),
            )
        else:
            return KnownValue(attr_value, source_node=node)

    def _visit_get_attribute_unbound(self, node, attr, root_value, on_error):
        method = root_value.get_method()
        if method is None:
            return UNRESOLVED_VALUE
        try:
            getattr(method, attr)
        except AttributeError:
            return on_error(method)
        return UnboundMethodValue(
            root_value.attr_name, root_value.typ, secondary_attr_name=attr
        )

    def _visit_set_attribute(self, node, root_value):
        if isinstance(root_value, (TypedValue, SubclassValue)):
            typ = root_value.typ
        elif isinstance(root_value, KnownValue):
            typ = type(root_value.val)
        else:
            return None

        if self.scope.scope_type() == ScopeType.function_scope:
            composite = self.varname_for_constraint(node)
            if composite:
                self.scope.set(composite, self.being_assigned, node, self.state)

        self._record_type_attr_set(typ, node.attr, node, self.being_assigned)

    def varname_for_constraint(self, node):
        """Given a node, returns a variable name that could be used in a local scope."""
        if isinstance(node, ast.Attribute):
            attribute_path = self._get_attribute_path(node)
            if attribute_path:
                attributes = tuple(attribute_path[1:])
                return CompositeVariable(attribute_path[0], attributes)
            else:
                return None
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return None

    def _should_ignore_val(self, node):
        if node is not None:
            path = self._get_attribute_path(node)
            if path is not None:
                for ignored_path in self.config.IGNORED_PATHS:
                    if path[: len(ignored_path)] == ignored_path:
                        return True
                if path[-1] in self.config.IGNORED_END_OF_REFERENCE:
                    self.log(logging.INFO, "Ignoring end of reference", path)
                    return True
        return False

    def _should_ignore_type(self, typ):
        """Types for which we do not check whether they support the actions we take on them."""
        return typ in self.config.IGNORED_TYPES

    # Call nodes

    def visit_keyword(self, node):
        return (node.arg, self.visit(node.value))

    def visit_Call(self, node):
        """Call nodes represent function or other calls.

        We try to resolve the callee node into a value and then check whether its signature
        is compatible with the arguments that are passed to it.

        """
        val, _ = self.constraint_from_call(node)
        return val

    def constraint_from_call(self, node):
        callee_wrapped = self.visit(node.func)
        args = self._generic_visit_list(node.args)
        keywords = self._generic_visit_list(node.keywords)

        # these don't exist in py3
        if getattr(node, "starargs", None) is not None:
            self.visit(node.starargs)
        if getattr(node, "kwargs", None) is not None:
            self.visit(node.kwargs)
        if six.PY2:
            has_args_kwargs = node.starargs is not None or node.kwargs is not None
        else:
            has_args_kwargs = any(
                isinstance(arg, ast.Starred) for arg in node.args
            ) or any(kw.arg is None for kw in node.keywords)

        return_value, constraint = self._get_argspec_and_check_call(
            node, callee_wrapped, args, keywords, has_args_kwargs
        )

        if self._is_checking():
            self.yield_checker.record_call(callee_wrapped, node)
            self.asynq_checker.check_call(callee_wrapped, node)

        callee_val = None
        if isinstance(callee_wrapped, UnboundMethodValue):
            callee_val = callee_wrapped.get_method()
        elif isinstance(callee_wrapped, KnownValue):
            callee_val = callee_wrapped.val

        if callee_val is not None:
            if self.collector is not None:
                caller = (
                    self.current_function
                    if self.current_function is not None
                    else self.module
                )
                self.collector.record_call(caller, callee_val)

            if (
                (return_value is UNRESOLVED_VALUE or return_value == KnownValue(None))
                and inspect.isclass(callee_val)
                and not self._safe_in(callee_val, self.config.IGNORED_CALLEES)
            ):
                # if all arguments are KnownValues and the class is whitelisted, instantiate it
                if issubclass(
                    callee_val, self.config.CLASSES_SAFE_TO_INSTANTIATE
                ) and self._can_perform_call(node, args, keywords):
                    return_value = self._try_perform_call(
                        callee_val, node, args, keywords, TypedValue(callee_val)
                    )
                else:
                    # calls to types result in values of that type
                    return_value = TypedValue(callee_val)
            elif self._safe_in(
                callee_val, self.config.FUNCTIONS_SAFE_TO_CALL
            ) and self._can_perform_call(node, args, keywords):
                return_value = self._try_perform_call(
                    callee_val, node, args, keywords, return_value
                )

        return return_value, constraint

    def _safe_in(self, item, collection):
        # Workaround against mock objects sometimes throwing ValueError if you compare them,
        # and against objects throwing other kinds of errors if you use in.
        try:
            return item in collection
        except Exception:
            return False

    def _can_perform_call(self, node, args, keywords):
        """Returns whether all of the arguments were inferred successfully."""
        return (
            getattr(node, "starargs", None) is None
            and getattr(node, "kwargs", None) is None
            and all(isinstance(arg, KnownValue) for arg in args)
            and all(isinstance(arg, KnownValue) for _, arg in keywords)
        )

    def _try_perform_call(self, callee_val, node, args, keywords, fallback_return):
        """Tries to call callee_val with the given arguments.

        Falls back to fallback_return and emits an error if the call fails.

        """
        unwrapped_args = [arg.val for arg in args]
        unwrapped_kwargs = {key: value.val for key, value in keywords}
        try:
            value = callee_val(*unwrapped_args, **unwrapped_kwargs)
        except Exception as e:
            message = "Error in {}: {}".format(safe_str(callee_val), safe_str(e))
            self._show_error_if_checking(node, message, ErrorCode.incompatible_call)
            return fallback_return
        else:
            return KnownValue(value)

    def _get_argspec_and_check_call(
        self,
        node,
        callee_wrapped,
        args,
        keywords=[],
        has_args_kwargs=False,
        allow_call=False,
    ):
        if not isinstance(callee_wrapped, (KnownValue, TypedValue, UnboundMethodValue)):
            return UNRESOLVED_VALUE, NULL_CONSTRAINT

        if isinstance(callee_wrapped, KnownValue) and any(
            callee_wrapped.val is ignored for ignored in self.config.IGNORED_CALLEES
        ):
            self.log(logging.INFO, "Ignoring callee", callee_wrapped)
            return UNRESOLVED_VALUE, NULL_CONSTRAINT

        extended_argspec = self._get_argspec_from_value(callee_wrapped, node)

        if extended_argspec is None:
            return_value = UNRESOLVED_VALUE
            constraint = NULL_CONSTRAINT

        elif has_args_kwargs:
            # TODO(jelle): handle this better
            return_value = UNRESOLVED_VALUE
            constraint = NULL_CONSTRAINT

        else:
            if isinstance(callee_wrapped, UnboundMethodValue):
                args = [TypedValue(callee_wrapped.typ)] + args

            if self._is_checking():
                (
                    return_value,
                    constraint,
                    no_return_unless,
                ) = extended_argspec.check_call(args, keywords, self, node)
            else:
                with self.catch_errors():
                    (
                        return_value,
                        constraint,
                        no_return_unless,
                    ) = extended_argspec.check_call(args, keywords, self, node)

            if no_return_unless is not NULL_CONSTRAINT:
                self.add_constraint(node, no_return_unless)

            if (
                not extended_argspec.has_return_value()
                and id(extended_argspec) in self._argspec_to_retval
            ):
                return_value = self._argspec_to_retval[id(extended_argspec)]

        if (
            allow_call
            and isinstance(callee_wrapped, KnownValue)
            and all(isinstance(arg, KnownValue) for arg in args)
            and all(isinstance(value, KnownValue) for key, value in keywords)
        ):
            try:
                return_value = KnownValue(
                    callee_wrapped.val(
                        *[arg.val for arg in args],
                        **{key: value.val for key, value in keywords}
                    )
                )
            except Exception as e:
                self.log(logging.INFO, "exception calling", (callee_wrapped, e))

        # for .asynq functions, we use the argspec for the underlying function, but that means
        # that the return value is not wrapped in AsyncTask, so we do that manually here
        if isinstance(callee_wrapped, KnownValue) and is_dot_asynq_function(
            callee_wrapped.val
        ):
            async_fn = callee_wrapped.val.__self__
            return (
                AsyncTaskIncompleteValue(_get_task_cls(async_fn), return_value),
                constraint,
            )
        elif isinstance(
            callee_wrapped, UnboundMethodValue
        ) and callee_wrapped.secondary_attr_name in ("async", "asynq"):
            async_fn = getattr(callee_wrapped.typ, callee_wrapped.attr_name)
            return (
                AsyncTaskIncompleteValue(_get_task_cls(async_fn), return_value),
                constraint,
            )
        elif isinstance(callee_wrapped, UnboundMethodValue) and asynq.is_pure_async_fn(
            callee_wrapped.get_method()
        ):
            return return_value, constraint
        else:
            if (
                return_value is UNRESOLVED_VALUE
                and isinstance(callee_wrapped, KnownValue)
                and asynq.is_pure_async_fn(callee_wrapped.val)
            ):
                return TypedValue(_get_task_cls(callee_wrapped.val)), constraint
            return return_value, constraint

    def _get_argspec_from_value(self, callee_wrapped, node):
        if isinstance(callee_wrapped, KnownValue):
            try:
                name = callee_wrapped.val.__name__
            # Ideally this would just catch AttributeError, but some objects raise
            # other exceptions from __getattr__.
            except Exception:
                name = None
            return self._get_argspec(callee_wrapped.val, node, name=name)
        elif isinstance(callee_wrapped, UnboundMethodValue):
            method = callee_wrapped.get_method()
            if method is not None:
                return self._get_argspec(method, node, name=callee_wrapped.attr_name)
        elif isinstance(callee_wrapped, TypedValue):
            typ = callee_wrapped.typ
            name = typ.__name__
            if not hasattr(typ, "__call__") or (
                getattr(typ.__call__, "__objclass__", None) is type
                and not issubclass(typ, type)
            ):
                self._show_error_if_checking(
                    node,
                    "Object of type %r is not callable" % (typ,),
                    ErrorCode.not_callable,
                )
                return None
            call_fn = typ.__call__
            if hasattr(call_fn, "__func__"):  # py2
                call_fn = call_fn.__func__
            argspec = self._get_argspec(call_fn, node, name=name)
            if argspec is None:
                return None
            return BoundMethodArgSpecWrapper(argspec, callee_wrapped)
        return None

    def _get_argspec(self, obj, node, name=None):
        """Given a Python object obj retrieved from node, try to get its argspec."""
        try:
            return self.arg_spec_cache.get_argspec(obj, name=name, logger=self.log)
        except TypeError as e:
            self._show_error_if_checking(node, e, ErrorCode.not_callable)
            return None

    # Attribute checking

    def _record_class_examined(self, cls):
        if self.attribute_checker is not None:
            self.attribute_checker.record_class_examined(cls)

    def _record_type_has_dynamic_attrs(self, typ):
        if self.attribute_checker is not None:
            self.attribute_checker.record_type_has_dynamic_attrs(typ)

    def _record_type_attr_set(self, typ, attr_name, node, value):
        if self.attribute_checker is not None:
            self.attribute_checker.record_attribute_set(typ, attr_name, node, value)

    def _record_type_attr_read(self, typ, attr_name, node):
        if self.attribute_checker is not None:
            self.attribute_checker.record_attribute_read(typ, attr_name, node, self)

    def _maybe_get_attr_value(self, typ, attr_name):
        if self.attribute_checker is not None:
            return self.attribute_checker.get_attribute_value(typ, attr_name)
        else:
            return UNRESOLVED_VALUE

    # Finding unused objects

    def _maybe_record_usage(self, value):
        if self.unused_finder is None:
            return

        # in this case class isn't available
        if self.scope.scope_type() == ScopeType.function_scope and self._is_checking():
            return

        # exclude calls within a class (probably in super calls)
        if value is self.current_class:
            return

        inner = self.config.unwrap_cls(value)
        if inner is self.current_class:
            return

        self.unused_finder.record(value)

    @classmethod
    def _get_argument_parser(cls):
        parser = super(NameCheckVisitor, cls)._get_argument_parser()
        parser.add_argument(
            "--find-unused",
            action="store_true",
            default=False,
            help="Find unused functions and classes",
        )
        parser.add_argument(
            "--find-unused-attributes",
            action="store_true",
            default=False,
            help="Find unused class attributes",
        )
        # TODO(jelle): this has false positives (e.g. from ... import)
        parser.add_argument(
            "--include-modules",
            action="store_true",
            default=False,
            help="Include modules in the unused objects check.",
        )
        return parser

    @classmethod
    def is_enabled_by_default(cls, code):
        if code in DISABLED_BY_DEFAULT:
            return code in cls.config.ENABLED_ERRORS
        else:
            return code not in cls.config.DISABLED_ERRORS

    @classmethod
    def get_description_for_error_code(cls, error_code):
        return ERROR_DESCRIPTION[error_code]

    @classmethod
    def get_default_modules(cls):
        return (cls.config.DEFAULT_BASE_MODULE,)

    @classmethod
    def get_default_directories(cls):
        return cls.config.DEFAULT_DIRS

    @classmethod
    def _run_on_files_or_all(
        cls,
        find_unused=False,
        include_modules=False,
        settings=None,
        find_unused_attributes=False,
        **kwargs
    ):
        attribute_checker_enabled = settings[ErrorCode.attribute_is_never_set]
        if "arg_spec_cache" not in kwargs:
            kwargs["arg_spec_cache"] = ArgSpecCache(cls.config)
        with ClassAttributeChecker(
            cls.config,
            enabled=attribute_checker_enabled,
            should_check_unused_attributes=find_unused_attributes,
        ) as attribute_checker, UnusedObjectFinder(
            cls.config, enabled=find_unused, include_modules=include_modules
        ) as unused_finder:
            all_failures = super(NameCheckVisitor, cls)._run_on_files_or_all(
                attribute_checker=attribute_checker,
                unused_finder=unused_finder,
                settings=settings,
                **kwargs
            )
        if attribute_checker is not None:
            all_failures += attribute_checker.all_failures
        return all_failures

    @classmethod
    def check_all_files(cls, *args, **kwargs):
        if "arg_spec_cache" not in kwargs:
            kwargs["arg_spec_cache"] = ArgSpecCache(cls.config)
        return super(NameCheckVisitor, cls).check_all_files(*args, **kwargs)

    @classmethod
    def _should_ignore_module(cls, module_name):
        """Override this to ignore some modules."""
        # exclude test modules for now to avoid spurious failures
        # TODO(jelle): enable for test modules too
        return module_name.split(".")[-1].startswith("test")

    @classmethod
    def check_file_in_worker(cls, filename, attribute_checker=None, **kwargs):
        failures = cls.check_file(
            filename, attribute_checker=attribute_checker, **kwargs
        )
        return failures, attribute_checker

    @classmethod
    def merge_extra_data(cls, extra_data, attribute_checker=None, **kwargs):
        if attribute_checker is None:
            return
        for checker in extra_data:
            if checker is None:
                continue
            for serialized, attrs in checker.attributes_read.items():
                attribute_checker.attributes_read[serialized] += attrs
            for serialized, attrs in checker.attributes_set.items():
                attribute_checker.attributes_set[serialized] |= attrs
            for serialized, attrs in checker.attribute_values.items():
                for attr_name, value in attrs.items():
                    attribute_checker.merge_attribute_value(
                        serialized, attr_name, value
                    )
            attribute_checker.modules_examined |= checker.modules_examined
            attribute_checker.classes_examined |= checker.modules_examined
            attribute_checker.types_with_dynamic_attrs |= (
                checker.types_with_dynamic_attrs
            )
            attribute_checker.filename_to_visitor.update(checker.filename_to_visitor)


def _get_task_cls(fn):
    """Returns the task class for an async function."""

    if hasattr(fn, "task_cls"):
        cls = fn.task_cls
    elif hasattr(fn, "decorator") and hasattr(fn.decorator, "task_cls"):
        cls = fn.decorator.task_cls
    else:
        cls = asynq.AsyncTask

    if cls is None:  # @async_proxy()
        return asynq.FutureBase
    else:
        return cls


def _all_names_unused(elts, unused_name_nodes):
    """Given the left-hand side of an assignment, returns whether all names assigned to are unused.

    elts is a list of assignment nodes, which may contain nested lists or tuples. unused_name_nodes
    is a list of Name nodes corresponding to unused variables.

    """
    for elt in elts:
        if isinstance(elt, (ast.List, ast.Tuple)):
            if not _all_names_unused(elt.elts, unused_name_nodes):
                return False
        if elt not in unused_name_nodes:
            return False
    return True


def _contains_node(elts, node):
    """Given a list of assignment targets (elts), return whether it contains the given Name node."""
    for elt in elts:
        if isinstance(elt, (ast.List, ast.Tuple)):
            if _contains_node(elt.elts, node):
                return True
        if elt is node:
            return True
    return False


def _static_hasattr(value, attr):
    """Returns whether this value has the given attribute, ignoring __getattr__ overrides."""
    try:
        object.__getattribute__(value, attr)
    except AttributeError:
        return False
    else:
        return True


def _safe_getattr(value, attr, default):
    """Returns whether this value has the given attribute, ignoring exceptions."""
    try:
        return getattr(value, attr)
    except Exception:
        return default


def _is_coroutine_function(obj):
    try:
        return inspect2.iscoroutinefunction(obj)
    except AttributeError:
        # This can happen to cached classmethods.
        return False


def _has_annotation_for_attr(typ, attr):
    try:
        return attr in typ.__annotations__
    except Exception:
        # __annotations__ doesn't exist or isn't a dict
        return False


def _get_attrs_attribute(typ, attr):
    try:
        if hasattr(typ, "__attrs_attrs__"):
            for attr_attr in typ.__attrs_attrs__:
                if attr_attr.name == attr:
                    if attr_attr.type is not None:
                        return type_from_runtime(attr_attr.type)
                    else:
                        return UNRESOLVED_VALUE
    except Exception:
        # Guard against silly objects throwing exceptions on hasattr()
        # or similar shenanigans.
        pass
    return None
