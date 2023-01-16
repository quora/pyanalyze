"""

The core of the pyanalyze type checker.

:class:`NameCheckVisitor` is the AST visitor that powers pyanalyze's
type inference. It is the central object that invokes other parts of
the system.

"""
import abc
import ast
import asyncio
import collections
import collections.abc
import contextlib
import enum
import itertools
import logging
import operator
import os
import os.path
import pickle
import sys
import traceback
import types
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Container,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import asynq
import qcore
import typeshed_client
from typing_extensions import Annotated, Protocol

from . import attributes, format_strings, importer, node_visitor, type_evaluation
from .analysis_lib import get_attribute_path
from .annotations import (
    is_context_manager_type,
    is_instance_of_typing_name,
    is_typing_name,
    SyntheticEvaluator,
    type_from_annotations,
    type_from_value,
)
from .arg_spec import ArgSpecCache, IgnoredCallees, is_dot_asynq_function, UnwrapClass
from .asynq_checker import AsynqChecker
from .boolability import Boolability, get_boolability
from .checker import Checker, CheckerAttrContext
from .error_code import ERROR_DESCRIPTION, ErrorCode
from .extensions import assert_error, ParameterTypeGuard, patch_typing_overload
from .find_unused import UnusedObjectFinder, used
from .functions import (
    AsyncFunctionKind,
    compute_function_info,
    compute_value_of_function,
    FunctionDefNode,
    FunctionInfo,
    FunctionNode,
    FunctionResult,
    GeneratorValue,
    IMPLICIT_CLASSMETHODS,
    ReturnT,
    SendT,
    YieldT,
)
from .options import (
    add_arguments,
    BooleanOption,
    ConcatenatedOption,
    ConfigOption,
    IntegerOption,
    InvalidConfigOption,
    Options,
    PyObjectSequenceOption,
    StringSequenceOption,
)
from .patma import PatmaVisitor
from .predicates import EqualsPredicate
from .reexport import ImplicitReexportTracker
from .safe import (
    is_dataclass_type,
    is_hashable,
    safe_getattr,
    safe_hasattr,
    safe_isinstance,
    safe_issubclass,
)
from .shared_options import EnforceNoUnused, ImportPaths, Paths
from .signature import (
    ANY_SIGNATURE,
    ARGS,
    ConcreteSignature,
    KWARGS,
    MaybeSignature,
    OverloadedSignature,
    Signature,
    SigParameter,
)
from .stacked_scopes import (
    AbstractConstraint,
    AndConstraint,
    annotate_with_constraint,
    Composite,
    CompositeIndex,
    constrain_value,
    Constraint,
    ConstraintType,
    EMPTY_ORIGIN,
    EquivalentConstraint,
    extract_constraints,
    FALSY_CONSTRAINT,
    FunctionScope,
    LEAVES_LOOP,
    LEAVES_SCOPE,
    NULL_CONSTRAINT,
    OrConstraint,
    PredicateProvider,
    ScopeType,
    StackedScopes,
    SubScope,
    TRUTHY_CONSTRAINT,
    Varname,
    VarnameOrigin,
    VarnameWithOrigin,
    VisitorState,
)
from .suggested_type import (
    CallArgs,
    display_suggested_type,
    prepare_type,
    should_suggest_type,
)
from .type_object import get_mro, TypeObject
from .value import (
    AlwaysPresentExtension,
    DeprecatedExtension,
    SkipDeprecatedExtension,
    annotate_value,
    AnnotatedValue,
    AnySource,
    AnyValue,
    AssertErrorExtension,
    AsyncTaskIncompleteValue,
    CallableValue,
    CanAssign,
    CanAssignError,
    check_hashability,
    concrete_values_from_iterable,
    ConstraintExtension,
    DictIncompleteValue,
    flatten_values,
    GenericBases,
    GenericValue,
    get_tv_map,
    is_iterable,
    is_union,
    KnownValue,
    kv_pairs_from_mapping,
    KVPair,
    make_coro_type,
    MultiValuedValue,
    NO_RETURN_VALUE,
    NoReturnConstraintExtension,
    ReferencingValue,
    replace_known_sequence_value,
    SequenceValue,
    set_self,
    SubclassValue,
    TypedValue,
    TypeVarValue,
    unannotate_value,
    UnboundMethodValue,
    UNINITIALIZED_VALUE,
    unite_and_simplify,
    unite_values,
    unpack_values,
    Value,
    VOID,
)
from .yield_checker import YieldChecker

try:
    from ast import NamedExpr
except ImportError:
    NamedExpr = Any  # 3.7 and lower

try:
    from ast import Match
except ImportError:
    # 3.9 and lower
    Match = Any

try:
    from ast import TryStar
    from builtins import BaseExceptionGroup, ExceptionGroup
except ImportError:
    # 3.10 and lower
    TryStar = Any

    class BaseExceptionGroup:
        pass

    class ExceptionGroup:
        pass


T = TypeVar("T")
U = TypeVar("U")
AwaitableValue = GenericValue(collections.abc.Awaitable, [TypeVarValue(T)])
KnownNone = KnownValue(None)
ExceptionValue = TypedValue(BaseException) | SubclassValue(TypedValue(BaseException))
ExceptionOrNone = ExceptionValue | KnownNone


BINARY_OPERATION_TO_DESCRIPTION_AND_METHOD = {
    ast.Add: ("addition", "__add__", "__iadd__", "__radd__"),
    ast.Sub: ("subtraction", "__sub__", "__isub__", "__rsub__"),
    ast.Mult: ("multiplication", "__mul__", "__imul__", "__rmul__"),
    ast.Div: ("division", "__truediv__", "__itruediv__", "__rtruediv__"),
    ast.Mod: ("modulo", "__mod__", "__imod__", "__rmod__"),
    ast.Pow: ("exponentiation", "__pow__", "__ipow__", "__rpow__"),
    ast.LShift: ("left-shifting", "__lshift__", "__ilshift__", "__rlshift__"),
    ast.RShift: ("right-shifting", "__rshift__", "__irshift__", "__rrshift__"),
    ast.BitOr: ("bitwise OR", "__or__", "__ior__", "__ror__"),
    ast.BitXor: ("bitwise XOR", "__xor__", "__ixor__", "__rxor__"),
    ast.BitAnd: ("bitwise AND", "__and__", "__iand__", "__rand__"),
    ast.FloorDiv: ("floor division", "__floordiv__", "__ifloordiv__", "__rfloordiv__"),
    ast.MatMult: ("matrix multiplication", "__matmul__", "__imatmul__", "__rmatmul__"),
    ast.Eq: ("equality", "__eq__", None, "__eq__"),
    ast.NotEq: ("inequality", "__ne__", None, "__ne__"),
    ast.Lt: ("less than", "__lt__", None, "__gt__"),
    ast.LtE: ("less than or equal", "__le__", None, "__ge__"),
    ast.Gt: ("greater than", "__gt__", None, "__lt__"),
    ast.GtE: ("greater than or equal", "__ge__", None, "__le__"),
    ast.In: ("contains", "__contains__", None, None),
    ast.NotIn: ("contains", "__contains__", None, None),
}

# Certain special methods are expected to return NotImplemented if they
# can't handle a particular argument, so that the interpreter can
# try some other call. To support thiis, such methods are allowed to
# return NotImplemented, even if their return annotation says otherwise.
# This is pieced together from the CPython source code, including:
# - Methods defined as SLOT1BIN in Objects/typeobject.c
# - Objects/abstract.c also does the binops
# - Rich comparison in object.c and typeobject.c
METHODS_ALLOWING_NOTIMPLEMENTED = {
    *[
        method
        for method in itertools.chain.from_iterable(
            data[1:] for data in BINARY_OPERATION_TO_DESCRIPTION_AND_METHOD.values()
        )
        if method is not None
    ],
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__length_hint__",  # Objects/abstract.c
    "__subclasshook__",  # Modules/_abc.c
}

UNARY_OPERATION_TO_DESCRIPTION_AND_METHOD = {
    ast.Invert: ("inversion", "__invert__"),
    ast.UAdd: ("unary positive", "__pos__"),
    ast.USub: ("unary negation", "__neg__"),
}


def _in(a: object, b: Container[object]) -> bool:
    return operator.contains(b, a)


def _not_in(a: object, b: Container[object]) -> bool:
    return not operator.contains(b, a)


COMPARATOR_TO_OPERATOR = {
    ast.Eq: (operator.eq, operator.ne),
    ast.NotEq: (operator.ne, operator.eq),
    ast.Lt: (operator.lt, operator.ge),
    ast.LtE: (operator.le, operator.gt),
    ast.Gt: (operator.gt, operator.le),
    ast.GtE: (operator.ge, operator.lt),
    ast.Is: (operator.is_, operator.is_not),
    ast.IsNot: (operator.is_not, operator.is_),
    ast.In: (_in, _not_in),
    ast.NotIn: (_not_in, _in),
}

SAFE_DECORATORS_FOR_ARGSPEC_TO_RETVAL = [KnownValue(asynq.asynq), KnownValue(property)]
if sys.version_info < (3, 11):
    # static analysis: ignore[undefined_attribute]
    SAFE_DECORATORS_FOR_ARGSPEC_TO_RETVAL.append(KnownValue(asyncio.coroutine))


class CustomContextManager(Protocol[T, U]):
    def __enter__(self) -> T:
        raise NotImplementedError

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[types.TracebackType],
    ) -> U:
        raise NotImplementedError


class AsyncCustomContextManager(Protocol[T, U]):
    async def __aenter__(self) -> T:
        raise NotImplementedError

    async def __aexit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[types.TracebackType],
    ) -> U:
        raise NotImplementedError


@dataclass
class _StarredValue(Value):
    """Helper Value to represent the result of "*x".

    Should not escape this file.

    """

    value: Value
    node: ast.AST

    def __init__(self, value: Value, node: ast.AST) -> None:
        self.value = value
        self.node = node


@dataclass(init=False)
class _AttrContext(CheckerAttrContext):
    visitor: "NameCheckVisitor"
    node: Optional[ast.AST]
    ignore_none: bool = False
    record_reads: bool = True

    # Needs to be implemented explicitly to work around Cython limitations
    def __init__(
        self,
        root_composite: Composite,
        attr: str,
        visitor: "NameCheckVisitor",
        *,
        node: Optional[ast.AST],
        ignore_none: bool = False,
        skip_mro: bool = False,
        skip_unwrap: bool = False,
        prefer_typeshed: bool = False,
        record_reads: bool = True,
    ) -> None:
        super().__init__(
            root_composite,
            attr,
            visitor.options,
            skip_mro=skip_mro,
            skip_unwrap=skip_unwrap,
            prefer_typeshed=prefer_typeshed,
            checker=visitor.checker,
        )
        self.node = node
        self.visitor = visitor
        self.ignore_none = ignore_none
        self.record_reads = record_reads

    def record_usage(self, obj: object, val: Value) -> None:
        self.visitor._maybe_record_usage(obj, self.attr, val)

    def record_attr_read(self, obj: type) -> None:
        if self.record_reads and self.node is not None:
            self.visitor._record_type_attr_read(obj, self.attr, self.node)

    def get_property_type_from_argspec(self, obj: property) -> Value:
        if obj.fget is None:
            return UNINITIALIZED_VALUE

        getter = set_self(KnownValue(obj.fget), self.root_composite.value)
        return self.visitor.check_call(self.node, getter, [self.root_composite])

    def should_ignore_none_attributes(self) -> bool:
        return self.ignore_none


class ComprehensionLengthInferenceLimit(IntegerOption):
    """If we iterate over something longer than this, we don't try to infer precise
    types for comprehensions. Increasing this can hurt performance."""

    default_value = 25
    name = "comprehension_length_inference_limit"


class UnionSimplificationLimit(IntegerOption):
    """We may simplify unions with more than this many values."""

    default_value = 100
    name = "union_simplification_limit"


class DisallowCallsToDunders(StringSequenceOption):
    """Set of dunder methods (e.g., '{"__lshift__"}') that pyanalyze is not allowed to call on
    objects."""

    name = "disallow_calls_to_dunders"


class ForLoopAlwaysEntered(BooleanOption):
    """If True, we assume that for loops are always entered at least once,
    which affects the potentially_undefined_name check. This will miss
    some bugs but also remove some annoying false positives."""

    name = "for_loop_always_entered"


class IgnoreNoneAttributes(BooleanOption):
    """If True, we ignore None when type checking attribute access on a Union
    type."""

    name = "ignore_none_attributes"


class UnimportableModules(StringSequenceOption):
    """Do not attempt to import these modules if they are imported within a function."""

    default_value = []
    name = "unimportable_modules"


class ExtraBuiltins(StringSequenceOption):
    """Even if these variables are undefined, no errors are shown."""

    name = "extra_builtins"
    default_value = ["__IPYTHON__"]  # special global defined in IPython


class IgnoredPaths(ConcatenatedOption[Sequence[str]]):
    """Attribute accesses on these do not result in errors."""

    name = "ignored_paths"
    default_value = ()

    # too complicated and this option isn't too useful anyway
    should_create_command_line_option = False

    @classmethod
    def parse(cls, data: object, source_path: Path) -> Sequence[Sequence[str]]:
        if not isinstance(data, (list, tuple)):
            raise InvalidConfigOption.from_parser(cls, "sequence", data)
        for sublist in data:
            if not isinstance(sublist, (list, tuple)):
                raise InvalidConfigOption.from_parser(cls, "sequence", sublist)
            for elt in sublist:
                if not isinstance(elt, str):
                    raise InvalidConfigOption.from_parser(cls, "string", elt)
        return data


class IgnoredEndOfReference(StringSequenceOption):
    """When these attributes are accessed but they don't exist, the error is ignored."""

    name = "ignored_end_of_reference"
    default_value = [
        # these are created by the mock module
        "call_count",
        "assert_has_calls",
        "reset_mock",
        "called",
        "assert_called_once",
        "assert_called_once_with",
        "assert_called_with",
        "count",
        "assert_any_call",
        "assert_not_called",
    ]


class IgnoredForIncompatibleOverride(StringSequenceOption):
    """These attributes are not checked for incompatible overrides."""

    name = "ignored_for_incompatible_overrides"
    default_value = ["__init__", "__eq__", "__ne__"]


class IgnoredUnusedAttributes(StringSequenceOption):
    """When these attributes are unused, they are not listed as such by the unused attribute
    finder."""

    name = "ignored_unused_attributes"
    default_value = [
        # ABCs
        "_abc_cache",
        "_abc_negative_cache",
        "__abstractmethods__",
        "_abc_negative_cache_version",
        "_abc_registry",
        # Python core
        "__module__",
        "__doc__",
        "__init__",
        "__dict__",
        "__weakref__",
        "__enter__",
        "__exit__",
        "__metaclass__",
    ]


class IgnoredUnusedClassAttributes(ConcatenatedOption[Tuple[type, Set[str]]]):
    """List of pairs of (class, set of attribute names). When these attribute names are seen as
    unused on a child or base class of the class, they are not listed."""

    name = "ignored_unused_class_attributes"
    default_value = []
    should_create_command_line_option = False  # too complicated

    @classmethod
    def parse(cls, data: object, source_path: Path) -> Sequence[Tuple[type, Set[str]]]:
        if not isinstance(data, (list, tuple)):
            raise InvalidConfigOption.from_parser(
                cls, "sequence of (type, [attribute]) pairs", data
            )
        final = []
        for elt in data:
            if not isinstance(elt, (list, tuple)) or len(elt) != 2:
                raise InvalidConfigOption.from_parser(
                    cls, "sequence of (type, [attribute]) pairs", elt
                )
            typ, attrs = elt
            try:
                obj = qcore.object_from_string(typ)
            except Exception:
                raise InvalidConfigOption.from_parser(
                    cls, "path to Python object", typ
                ) from None
            if not isinstance(attrs, (list, tuple)):
                raise InvalidConfigOption.from_parser(
                    cls, "sequence of attributes", attrs
                )
            for attr in attrs:
                if not isinstance(attr, str):
                    raise InvalidConfigOption.from_parser(cls, "attribute string", attr)
            final.append((obj, set(attrs)))
        return final


class CheckForDuplicateValues(PyObjectSequenceOption[type]):
    """For subclasses of these classes, we error if multiple attributes have the same
    value. This is used for the duplicate_enum check."""

    name = "check_for_duplicate_values"
    default_value = [enum.Enum]


class AllowDuplicateValues(PyObjectSequenceOption[type]):
    """For subclasses of these classes, we do not error if multiple attributes have the same
    value. This overrides CheckForDuplicateValues."""

    name = "allow_duplicate_values"
    default_value = []


def should_check_for_duplicate_values(cls: object, options: Options) -> bool:
    if not isinstance(cls, type):
        return False
    positive_list = tuple(options.get_value_for(CheckForDuplicateValues))
    if not safe_issubclass(cls, positive_list):
        return False
    negative_list = tuple(options.get_value_for(AllowDuplicateValues))
    if safe_issubclass(cls, negative_list):
        return False
    return True


class IgnoredTypesForAttributeChecking(PyObjectSequenceOption[type]):
    """Used in the check for object attributes that are accessed but not set. In general, the check
    will only alert about attributes that don't exist when it has visited all the base classes of
    the class with the possibly missing attribute. However, these classes are never going to be
    visited (since they're builtin), but they don't set any attributes that we rely on.
    """

    name = "ignored_types_for_attribute_checking"
    default_value = [object, abc.ABC]


class ClassAttributeChecker:
    """Helper class to keep track of attributes that are read and set on instances."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        should_check_unused_attributes: bool = False,
        should_serialize: bool = False,
        options: Options = Options.from_option_list(),
    ) -> None:
        self.options = options
        # we might not have examined all parent classes when looking for attributes set
        # we dump them here. incase the callers want to extend coverage.
        self.unexamined_base_classes = set()
        self.modules_examined = set()
        self.enabled = enabled
        self.should_check_unused_attributes = should_check_unused_attributes
        self.should_serialize = should_serialize
        self.all_failures = []
        self.types_with_dynamic_attrs = set()
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
            for typ in self.options.get_value_for(IgnoredTypesForAttributeChecking)
        }

    def __enter__(self) -> Optional["ClassAttributeChecker"]:
        if self.enabled:
            return self
        else:
            return None

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[types.TracebackType],
    ) -> None:
        if exc_type is None and self.enabled:
            self.check_attribute_reads()

            if self.should_check_unused_attributes:
                self.check_unused_attributes()

    def record_attribute_read(
        self, typ: type, attr_name: str, node: ast.AST, visitor: "NameCheckVisitor"
    ) -> None:
        """Records that attribute attr_name was accessed on type typ."""
        self.filename_to_visitor[visitor.filename] = visitor
        serialized = self.serialize_type(typ)
        if serialized is not None:
            self.attributes_read[serialized].append((attr_name, node, visitor.filename))

    def record_attribute_set(
        self, typ: type, attr_name: str, node: ast.AST, value: Value
    ) -> None:
        """Records that attribute attr_name was set on type typ."""
        serialized = self.serialize_type(typ)
        if serialized is None:
            return
        self.attributes_set[serialized].add(attr_name)
        self.merge_attribute_value(serialized, attr_name, value)

    def merge_attribute_value(
        self, serialized: object, attr_name: str, value: Value
    ) -> None:
        try:
            pickle.loads(pickle.dumps(value))
        except Exception:
            # If we can't serialize it, don't attempt to store it.
            value = AnyValue(AnySource.inference)
        scope = self.attribute_values[serialized]
        if attr_name not in scope:
            scope[attr_name] = value
        elif scope[attr_name] == value:
            pass
        else:
            scope[attr_name] = unite_values(scope[attr_name], value)

    def record_type_has_dynamic_attrs(self, typ: type) -> None:
        serialized = self.serialize_type(typ)
        if serialized is not None:
            self.types_with_dynamic_attrs.add(serialized)

    def record_class_examined(self, cls: type) -> None:
        """Records that we examined the attributes of class cls."""
        serialized = self.serialize_type(cls)
        if serialized is not None:
            self.classes_examined.add(serialized)

    def record_module_examined(self, module_name: str) -> None:
        self.modules_examined.add(module_name)

    def serialize_type(self, typ: type) -> object:
        """Serialize a type so it is pickleable.

        We do this to make it possible to pass ClassAttributeChecker objects around
        to parallel workers.

        """
        if not self.should_serialize:
            try:
                hash(typ)
            except Exception:
                return None  # ignore non-hashable types
            else:
                return typ
        if isinstance(typ, super):
            typ = typ.__self_class__
        if isinstance(safe_getattr(typ, "__module__", None), str) and isinstance(
            safe_getattr(typ, "__name__", None), str
        ):
            module = typ.__module__
            name = typ.__name__
            if module not in sys.modules:
                return None
            actual = safe_getattr(sys.modules[module], name, None)
            if UnwrapClass.unwrap(actual, self.options) is typ:
                return (module, name)
        return None

    def unserialize_type(self, serialized: Any) -> Optional[type]:
        if not self.should_serialize:
            return serialized
        module, name = serialized
        if module not in sys.modules:
            __import__(module)
        try:
            actual = getattr(sys.modules[module], name)
            return UnwrapClass.unwrap(actual, self.options)
        except AttributeError:
            # We've seen this happen when we import different modules under the same name.
            return None

    def get_attribute_value(self, typ: type, attr_name: str) -> Value:
        """Gets the current recorded value of the attribute."""
        for base_typ in get_mro(typ):
            serialized_base = self.serialize_type(base_typ)
            if serialized_base is None:
                continue
            value = self.attribute_values[serialized_base].get(attr_name)
            if value is not None:
                return value
        return AnyValue(AnySource.inference)

    def check_attribute_reads(self) -> None:
        """Checks that all recorded attribute reads refer to valid attributes.

        This is done by checking for each read whether the class has the attribute or whether any
        code sets the attribute on a class instance, among other conditions.

        """
        for serialized, attrs_read in sorted(
            self.attributes_read.items(), key=self._cls_sort
        ):
            typ = self.unserialize_type(serialized)
            if typ is None:
                continue
            # we setattr on it with an unresolved value, so we don't know what attributes this may
            # have
            if any(
                self.serialize_type(base_cls) in self.types_with_dynamic_attrs
                for base_cls in get_mro(typ)
            ):
                continue

            for attr_name, node, filename in sorted(
                attrs_read, key=lambda data: data[0]
            ):
                self._check_attribute_read(
                    typ, attr_name, node, self.filename_to_visitor[filename]
                )

    def check_unused_attributes(self) -> None:
        """Attempts to find attributes.

        This relies on comparing the set of attributes read on each class with the attributes in the
        class's ``__dict__``. It has many false positives and should be considered experimental.

        Some known causes of false positives:

        - Methods called in base classes of children (mixins)
        - Special methods like ``__eq__``
        - Insufficiently powerful type inference

        """
        all_attrs_read = collections.defaultdict(set)

        def _add_attrs(typ: Any, attr_names_read: Set[str]) -> None:
            if typ is None:
                return
            all_attrs_read[typ] |= attr_names_read
            for base_cls in typ.__bases__:
                all_attrs_read[base_cls] |= attr_names_read
            if isinstance(typ, type):
                for child_cls in qcore.inspection.get_subclass_tree(typ):
                    all_attrs_read[child_cls] |= attr_names_read

        for serialized, attrs_read in self.attributes_read.items():
            attr_names_read = {attr_name for attr_name, _, _ in attrs_read}
            _add_attrs(self.unserialize_type(serialized), attr_names_read)

        for typ, attrs in self.options.get_value_for(IgnoredUnusedClassAttributes):
            _add_attrs(typ, attrs)

        ignored = set(self.options.get_value_for(IgnoredUnusedAttributes))
        for typ, attrs_read in sorted(all_attrs_read.items(), key=self._cls_sort):
            if self.serialize_type(typ) not in self.classes_examined:
                continue
            existing_attrs = set(typ.__dict__.keys())
            for attr in existing_attrs - attrs_read - ignored:
                # server calls will always show up as unused here
                if safe_getattr(safe_getattr(typ, attr, None), "server_call", False):
                    continue
                print(f"Unused method: {typ!r}.{attr}")

    # sort by module + name in order to get errors in a reasonable order
    def _cls_sort(self, pair: Tuple[Any, Any]) -> Tuple[str, ...]:
        typ = pair[0]
        if hasattr(typ, "__name__") and isinstance(typ.__name__, str):
            return (str(typ.__module__), str(typ.__name__))
        else:
            return (str(typ), "")

    def _check_attribute_read(
        self, typ: type, attr_name: str, node: ast.AST, visitor: "NameCheckVisitor"
    ) -> None:
        # class itself has the attribute
        if hasattr(typ, attr_name):
            return
        # the attribute is in __annotations__, e.g. a dataclass
        if _has_annotation_for_attr(typ, attr_name) or attributes.get_attrs_attribute(
            typ,
            attributes.AttrContext(
                Composite(TypedValue(typ)),
                attr_name,
                visitor.options,
                skip_unwrap=False,
                skip_mro=False,
                prefer_typeshed=False,
            ),
        ):
            return

        # name mangling
        if attr_name.startswith("__") and hasattr(typ, f"_{typ.__name__}{attr_name}"):
            return

        # can't be sure whether it exists if class has __getattr__
        if hasattr(typ, "__getattr__") or (
            typ.__getattribute__ is not object.__getattribute__
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
        for base_cls in get_mro(typ):
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
            for base_cls in get_mro(child_cls):
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
            f"Attribute {attr_name} of type {typ} probably does not exist",
            ErrorCode.attribute_is_never_set,
        )
        # message can be None if the error is intercepted by error code settings or ignore
        # directives
        if message is not None:
            self.all_failures.append(message)

    def _should_reject_unexamined(self, base_cls: type) -> bool:
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


_AstType = Union[Type[ast.AST], Tuple[Type[ast.AST], ...]]


class StackedContexts:
    """Object to keep track of a stack of states.

    This is used to indicate all the AST node types that are parents of the node being examined.

    """

    contexts: List[ast.AST]

    def __init__(self) -> None:
        self.contexts = []

    def includes(self, typ: _AstType) -> bool:
        return any(isinstance(val, typ) for val in self.contexts)

    def nth_parent(self, n: int) -> Optional[ast.AST]:
        return self.contexts[-n] if len(self.contexts) >= n else None

    def nearest_enclosing(self, typ: _AstType) -> Optional[ast.AST]:
        for node in reversed(self.contexts):
            if isinstance(node, typ):
                return node
        return None

    @contextlib.contextmanager
    def add(self, value: ast.AST) -> Iterator[None]:
        """Context manager to add a context to the stack."""
        self.contexts.append(value)
        try:
            yield
        finally:
            self.contexts.pop()


@used  # exposed as an API
class CallSiteCollector:
    """Class to record function calls with their origin."""

    def __init__(self) -> None:
        self.map = collections.defaultdict(list)

    def record_call(self, caller: object, callee: object) -> None:
        try:
            self.map[callee].append(caller)
        except TypeError:
            # Unhashable callee. This is mostly calls to bound versions of list.append. We could get
            # the unbound method, but that doesn't seem very useful, so we just ignore it.
            pass


class NameCheckVisitor(node_visitor.ReplacingNodeVisitor):
    """Visitor class that infers the type and value of Python objects and detects errors.
    """

    error_code_enum = ErrorCode
    config_filename: ClassVar[Optional[str]] = None
    """Path (relative to this class's file) to a pyproject.toml config file."""

    _argspec_to_retval: Dict[int, Tuple[Value, MaybeSignature]]
    _has_used_any_match: bool
    _method_cache: Dict[Type[ast.AST], Callable[[Any], Optional[Value]]]
    _name_node_to_statement: Optional[Dict[ast.AST, Optional[ast.AST]]]
    _should_exclude_any: bool
    _statement_types: Set[Type[ast.AST]]
    ann_assign_type: Optional[Tuple[Optional[Value], bool]]
    annotate: bool
    arg_spec_cache: ArgSpecCache
    async_kind: AsyncFunctionKind
    asynq_checker: AsynqChecker
    attribute_checker: ClassAttributeChecker
    being_assigned: Optional[Value]
    checker: Checker
    collector: Optional[CallSiteCollector]
    current_class: Optional[type]
    current_enum_members: Optional[Dict[object, str]]
    current_function: Optional[object]
    current_function_info: Optional[FunctionInfo]
    current_function_name: Optional[str]
    error_for_implicit_any: bool
    expected_return_value: Optional[Value]
    future_imports: Set[str]
    in_annotation: bool
    in_comprehension_body: bool
    in_union_decomposition: bool
    import_name_to_node: Dict[str, Union[ast.Import, ast.ImportFrom]]
    is_async_def: bool
    is_compiled: bool
    is_generator: bool
    match_subject: Composite
    module: Optional[types.ModuleType]
    node_context: StackedContexts
    options: Options
    reexport_tracker: ImplicitReexportTracker
    return_values: List[Optional[Value]]
    scopes: StackedScopes
    state: VisitorState
    unused_finder: UnusedObjectFinder
    yield_checker: YieldChecker

    def __init__(
        self,
        filename: str,
        contents: str,
        tree: ast.Module,
        *,
        settings: Optional[Mapping[ErrorCode, bool]] = None,
        fail_after_first: bool = False,
        verbosity: int = logging.CRITICAL,
        unused_finder: Optional[UnusedObjectFinder] = None,
        module: Optional[types.ModuleType] = None,
        attribute_checker: Optional[ClassAttributeChecker] = None,
        collector: Optional[CallSiteCollector] = None,
        annotate: bool = False,
        add_ignores: bool = False,
        checker: Checker,
    ) -> None:
        super().__init__(
            filename,
            contents,
            tree,
            settings,
            fail_after_first=fail_after_first,
            verbosity=verbosity,
            add_ignores=add_ignores,
        )
        self.checker = checker

        # State (to use in with qcore.override)
        self.state = VisitorState.collect_names
        # value currently being assigned
        self.being_assigned = AnyValue(AnySource.inference)
        self.ann_assign_type = None
        # current match target
        self.match_subject = Composite(AnyValue(AnySource.inference))
        # current class (for inferring the type of cls and self arguments)
        self.current_class = None
        self.current_function_name = None
        self.current_function_info = None

        # async
        self.async_kind = AsyncFunctionKind.non_async
        self.is_generator = False  # set to True if this function is a generator
        # if true, we annotate each node we visit with its inferred value
        self.annotate = annotate
        # true if we're in the body of a comprehension's loop
        self.in_comprehension_body = False
        self.options = checker.options

        if module is not None:
            self.module = module
            self.is_compiled = False
        else:
            self.module, self.is_compiled = self._load_module()

        if self.module is not None and hasattr(self.module, "__name__"):
            module_path = tuple(self.module.__name__.split("."))
            self.options = checker.options.for_module(module_path)

        # Data storage objects
        self.unused_finder = unused_finder
        self.attribute_checker = attribute_checker
        self.arg_spec_cache = checker.arg_spec_cache
        self.reexport_tracker = checker.reexport_tracker
        if (
            self.attribute_checker is not None
            and self.module is not None
            and not self.is_compiled
        ):
            self.attribute_checker.record_module_examined(self.module.__name__)

        self.scopes = build_stacked_scopes(
            self.module,
            simplification_limit=self.options.get_value_for(UnionSimplificationLimit),
        )
        self.node_context = StackedContexts()
        self.asynq_checker = AsynqChecker(
            self.options, self.module, self.show_error, self.log, self.replace_node
        )
        self.yield_checker = YieldChecker(self)
        self.current_function = None
        self.expected_return_value = None
        self.current_enum_members = None
        self.is_async_def = False
        self.in_annotation = False
        self.in_union_decomposition = False
        self.collector = collector
        self.import_name_to_node = {}
        self.future_imports = set()  # active future imports in this file
        self.return_values = []
        self.error_for_implicit_any = self.options.is_error_code_enabled(
            ErrorCode.implicit_any
        )

        self._name_node_to_statement = None
        # Cache the return values of functions within this file, so that we can use them to
        # infer types. Previously, we cached this globally, but that makes things non-
        # deterministic because we'll start depending on the order modules are checked.
        self._argspec_to_retval = {}
        self._method_cache = {}
        self._statement_types = set()
        self._has_used_any_match = False
        self._should_exclude_any = False
        self._fill_method_cache()

    def get_local_return_value(self, sig: MaybeSignature) -> Optional[Value]:
        val, saved_sig = self._argspec_to_retval.get(id(sig), (None, None))
        if sig is not saved_sig:
            return None
        return val

    def make_type_object(self, typ: Union[type, super, str]) -> TypeObject:
        return self.checker.make_type_object(typ)

    def can_assume_compatibility(self, left: TypeObject, right: TypeObject) -> bool:
        return self.checker.can_assume_compatibility(left, right)

    def assume_compatibility(
        self, left: TypeObject, right: TypeObject
    ) -> ContextManager[None]:
        return self.checker.assume_compatibility(left, right)

    def has_used_any_match(self) -> bool:
        """Whether Any was used to secure a match."""
        return self._has_used_any_match

    def record_any_used(self) -> None:
        """Record that Any was used to secure a match."""
        self._has_used_any_match = True

    def reset_any_used(self) -> ContextManager[None]:
        """Context that resets the value used by :meth:`has_used_any_match` and
        :meth:`record_any_match`."""
        return qcore.override(self, "_has_used_any_match", False)

    def set_exclude_any(self) -> ContextManager[None]:
        """Within this context, `Any` is compatible only with itself."""
        return qcore.override(self, "_should_exclude_any", True)

    def should_exclude_any(self) -> bool:
        """Whether Any should be compatible only with itself."""
        return self._should_exclude_any

    def get_generic_bases(
        self, typ: Union[type, str], generic_args: Sequence[Value] = ()
    ) -> GenericBases:
        return self.arg_spec_cache.get_generic_bases(typ, generic_args)

    def get_signature(
        self, obj: object, is_asynq: bool = False
    ) -> Optional[ConcreteSignature]:
        return self.checker.get_signature(obj, is_asynq=is_asynq)

    def __reduce_ex__(self, proto: object) -> object:
        # Only pickle the attributes needed to get error reporting working
        return self.__class__, (self.filename, self.contents, self.tree, self.settings)

    def _load_module(self) -> Tuple[Optional[types.ModuleType], bool]:
        """Sets the module_path and module for this file."""
        if not self.filename:
            return None, False
        self.log(logging.INFO, "Checking file", (self.filename, os.getpid()))
        import_paths = self.options.get_value_for(ImportPaths)

        try:
            return importer.load_module_from_file(
                self.filename, import_paths=[str(p) for p in import_paths]
            )
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            # don't re-raise the error, just proceed without a module object
            # this can happen with scripts that aren't intended to be imported
            if not self.has_file_level_ignore():
                if self.tree is not None and self.tree.body:
                    node = self.tree.body[0]
                else:
                    node = None
                failure = self.show_error(
                    node,
                    f"Failed to import {self.filename} due to {e!r}",
                    error_code=ErrorCode.import_failed,
                )
                if failure is not None:
                    # Don't print a traceback if the error was suppressed.
                    traceback.print_exc()
            return None, False

    def check(self, ignore_missing_module: bool = False) -> List[node_visitor.Failure]:
        """Run the visitor on this module."""
        start_time = qcore.utime()
        try:
            if self.is_compiled:
                # skip compiled (Cythonized) files because pyanalyze will misinterpret the
                # AST in some cases (for example, if a function was cdefed)
                return []
            if self.tree is None:
                return self.all_failures
            if self.module is None and not ignore_missing_module:
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
            if (
                self.module is not None
                and self.unused_finder is not None
                and not self.has_file_level_ignore()
            ):
                self.unused_finder.record_module_visited(self.module)
            if self.module is not None and self.module.__name__ is not None:
                self.reexport_tracker.record_module_completed(self.module.__name__)
        except node_visitor.VisitorError:
            raise
        except Exception as e:
            self.show_error(
                None,
                f"{traceback.format_exc()}\nInternal error: {e!r}",
                error_code=ErrorCode.internal_error,
            )
        # Recover memory used for the AST. We keep the visitor object around later in order
        # to show ClassAttributeChecker errors, but those don't need the full AST.
        self.tree = None
        self._lines.__cached_per_instance_cache__.clear()
        self._argspec_to_retval.clear()
        end_time = qcore.utime()
        message = f"{self.filename} took {(end_time - start_time) / qcore.SECOND:.2f} s"
        self.logger.log(logging.INFO, message)
        return self.all_failures

    def visit(self, node: ast.AST) -> Value:
        """Visit a node and return the :class:`pyanalyze.value.Value` corresponding
        to the node."""
        # inline self.node_context.add and the superclass's visit() for performance
        node_type = type(node)
        method = self._method_cache[node_type]
        self.node_context.contexts.append(node)
        try:
            # This part inlines ReplacingNodeVisitor.visit
            if node_type in self._statement_types:
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
                f"{traceback.format_exc()}\nInternal error: {e!r}",
                error_code=ErrorCode.internal_error,
            )
            ret = AnyValue(AnySource.error)
        finally:
            self.node_context.contexts.pop()
        if ret is None:
            ret = VOID
        if self.annotate:
            node.inferred_value = ret
        if self.error_for_implicit_any:
            for val in ret.walk_values():
                if isinstance(val, AnyValue) and val.source is not AnySource.explicit:
                    self._show_error_if_checking(
                        node,
                        f"Inferred value contains Any: {ret}",
                        ErrorCode.implicit_any,
                    )
        return ret

    def generic_visit(self, node: ast.AST) -> None:
        # Inlined version of ast.Visitor.generic_visit for performance.
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

    def _fill_method_cache(self) -> None:
        for typ in qcore.inspection.get_subclass_tree(ast.AST):
            method = "visit_" + typ.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[typ] = visitor
            if issubclass(typ, ast.stmt):
                self._statement_types.add(typ)

    def _is_collecting(self) -> bool:
        return self.state == VisitorState.collect_names

    def _is_checking(self) -> bool:
        return self.state == VisitorState.check_names

    def _show_error_if_checking(
        self,
        node: ast.AST,
        msg: Optional[str] = None,
        error_code: Optional[ErrorCode] = None,
        *,
        replacement: Optional[node_visitor.Replacement] = None,
        detail: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """We usually should show errors only in the check_names state to avoid duplicate errors.
        """
        if self._is_checking():
            self.show_error(
                node,
                msg,
                error_code=error_code,
                replacement=replacement,
                detail=detail,
                extra_metadata=extra_metadata,
            )

    def _set_name_in_scope(
        self,
        varname: str,
        node: ast.AST,
        value: Value = AnyValue(AnySource.inference),
        *,
        private: bool = False,
        lookup_node: object = None,
    ) -> Tuple[Value, VarnameOrigin]:
        if lookup_node is None:
            lookup_node = node

        current_scope = self.scopes.current_scope()
        if self.ann_assign_type is not None:
            expected_type, is_final = self.ann_assign_type
            if not current_scope.set_declared_type(
                varname, expected_type, is_final, node
            ):
                self._show_error_if_checking(
                    node, f"{varname} already declared", ErrorCode.already_declared
                )

        else:
            declared_type = current_scope.get_declared_type(varname)
            if declared_type is not None:
                can_assign = declared_type.can_assign(value, self)
                if isinstance(can_assign, CanAssignError):
                    self._show_error_if_checking(
                        node,
                        (
                            f"Incompatible assignment: expected {declared_type}, got"
                            f" {value}"
                        ),
                        error_code=ErrorCode.incompatible_assignment,
                        detail=can_assign.display(),
                    )
            if current_scope.is_final(varname):
                self._show_error_if_checking(
                    node,
                    f"Cannot assign to final name {varname}",
                    ErrorCode.incompatible_assignment,
                )

        scope_type = current_scope.scope_type
        if self.module is not None and scope_type == ScopeType.module_scope:
            if self.module.__name__ is not None and not private:
                self.reexport_tracker.record_exported_attribute(
                    self.module.__name__, varname
                )
            if varname in current_scope.variables:
                value, _ = current_scope.get_local(varname, lookup_node, self.state)
                return value, EMPTY_ORIGIN
        if scope_type == ScopeType.class_scope:
            self._check_for_incompatible_overrides(varname, node, value)
            self._check_for_class_variable_redefinition(varname, node)
        origin = current_scope.set(varname, value, lookup_node, self.state)
        return value, origin

    def _check_for_incompatible_overrides(
        self, varname: str, node: ast.AST, value: Value
    ) -> None:
        if self.current_class is None:
            return
        if varname in self.options.get_value_for(IgnoredForIncompatibleOverride):
            return
        for base_class in self.get_generic_bases(self.current_class):
            if base_class is self.current_class:
                continue
            base_class_value = TypedValue(base_class)
            ctx = _AttrContext(
                Composite(base_class_value),
                varname,
                self,
                node=node,
                skip_mro=True,
                skip_unwrap=True,
                record_reads=False,
            )
            base_value = attributes.get_attribute(ctx)
            can_assign = self._can_assign_to_base(base_value, value)
            if isinstance(can_assign, CanAssignError):
                error = CanAssignError(
                    children=[
                        CanAssignError(f"Base class: {self.display_value(base_value)}"),
                        CanAssignError(f"Child class: {self.display_value(value)}"),
                        can_assign,
                    ]
                )
                self._show_error_if_checking(
                    node,
                    f"Value of {varname} incompatible with base class {base_class}",
                    ErrorCode.incompatible_override,
                    detail=str(error),
                )

    def display_value(self, value: Value) -> str:
        return self.checker.display_value(value)

    def _can_assign_to_base(self, base_value: Value, child_value: Value) -> CanAssign:
        if base_value is UNINITIALIZED_VALUE:
            return {}
        if isinstance(base_value, KnownValue) and callable(base_value.val):
            base_sig = self.signature_from_value(base_value)
            if not isinstance(base_sig, (Signature, OverloadedSignature)):
                return {}
            child_sig = self.signature_from_value(child_value)
            if not isinstance(child_sig, (Signature, OverloadedSignature)):
                return CanAssignError(f"{child_value} is not callable")
            base_bound = base_sig.bind_self(ctx=self)
            if base_bound is None:
                return {}
            child_bound = child_sig.bind_self(ctx=self)
            if child_bound is None:
                return CanAssignError(f"{child_value} is missing a 'self' argument")
            return base_bound.can_assign(child_bound, self)
        return base_value.can_assign(child_value, self)

    def _check_for_class_variable_redefinition(
        self, varname: str, node: ast.AST
    ) -> None:
        if varname not in self.scopes.current_scope().variables:
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
            f"Name {varname} is already defined",
            error_code=ErrorCode.class_variable_redefinition,
        )

    def resolve_name(
        self,
        node: ast.Name,
        error_node: Optional[ast.AST] = None,
        suppress_errors: bool = False,
    ) -> Tuple[Value, VarnameOrigin]:
        """Resolves a Name node to a value.

        :param node: Node to resolve the name from
        :type node: ast.AST

        :param error_node: If given, this AST node is used instead of `node`
                           for displaying errors.
        :type error_node: Optional[ast.AST]

        :param suppress_errors: If True, do not produce errors if the name is
                                undefined.
        :type suppress_errors: bool

        """
        if error_node is None:
            error_node = node
        value, defining_scope, origin = self.scopes.get_with_scope(
            node.id, node, self.state
        )
        if defining_scope is not None:
            if defining_scope.scope_type in (
                ScopeType.module_scope,
                ScopeType.class_scope,
            ):
                if defining_scope.scope_object is not None:
                    self._maybe_record_usage(
                        defining_scope.scope_object, node.id, value
                    )
        if value is UNINITIALIZED_VALUE:
            if suppress_errors or node.id in self.options.get_value_for(ExtraBuiltins):
                self.log(logging.INFO, "ignoring undefined name", node.id)
            else:
                self._show_error_if_checking(
                    error_node, f"Undefined name: {node.id}", ErrorCode.undefined_name
                )
            return AnyValue(AnySource.error), origin
        if isinstance(value, MultiValuedValue):
            subvals = value.vals
        elif isinstance(value, AnnotatedValue) and isinstance(
            (value.value), MultiValuedValue
        ):
            subvals = value.value.vals
        else:
            subvals = None

        if subvals is not None:
            if any(subval is UNINITIALIZED_VALUE for subval in subvals):
                self._show_error_if_checking(
                    error_node,
                    f"{node.id} may be used uninitialized",
                    ErrorCode.possibly_undefined_name,
                )
                new_mvv = MultiValuedValue(
                    [
                        AnyValue(AnySource.error)
                        if subval is UNINITIALIZED_VALUE
                        else subval
                        for subval in subvals
                    ]
                )
                if isinstance(value, AnnotatedValue):
                    return AnnotatedValue(new_mvv, value.metadata), origin
                else:
                    return new_mvv, origin
        return value, origin

    def _get_first_import_node(self) -> ast.stmt:
        return min(self.import_name_to_node.values(), key=lambda node: node.lineno)

    def _generic_visit_list(self, lst: Iterable[ast.AST]) -> List[Value]:
        return [self.visit(node) for node in lst]

    def _is_write_ctx(self, ctx: ast.AST) -> bool:
        return isinstance(ctx, (ast.Store, ast.Param))

    def _is_read_ctx(self, ctx: ast.AST) -> bool:
        return isinstance(ctx, (ast.Load, ast.Del))

    @contextlib.contextmanager
    def _set_current_class(self, current_class: type) -> Iterator[None]:
        if should_check_for_duplicate_values(current_class, self.options):
            current_enum_members = {}
        else:
            current_enum_members = None
        with qcore.override(self, "current_class", current_class), qcore.override(
            self.asynq_checker, "current_class", current_class
        ), qcore.override(self, "current_enum_members", current_enum_members):
            yield

    def visit_ClassDef(self, node: ast.ClassDef) -> Value:
        self._generic_visit_list(node.decorator_list)
        self._generic_visit_list(node.bases)
        self._generic_visit_list(node.keywords)
        value = self._visit_class_and_get_value(node)
        value, _ = self._set_name_in_scope(node.name, node, value)
        return value

    def _get_class_object(self, node: ast.ClassDef) -> Value:
        if self.scopes.scope_type() == ScopeType.module_scope:
            return self.scopes.get(node.name, node, self.state)
        elif (
            self.scopes.scope_type() == ScopeType.class_scope
            and self.current_class is not None
            and hasattr(self.current_class, "__dict__")
        ):
            runtime_obj = self.current_class.__dict__.get(node.name)
            if isinstance(runtime_obj, type):
                return KnownValue(runtime_obj)
        return AnyValue(AnySource.inference)

    def _visit_class_and_get_value(self, node: ast.ClassDef) -> Value:
        if self._is_checking():
            cls_obj = self._get_class_object(node)

            module = self.module
            if isinstance(cls_obj, MultiValuedValue) and module is not None:
                # if there are multiple, see if there is only one that matches this module
                possible_values = [
                    val
                    for val in cls_obj.vals
                    if isinstance(val, KnownValue)
                    and isinstance(val.val, type)
                    and safe_getattr(val.val, "__module__", None) == module.__name__
                ]
                if len(possible_values) == 1:
                    cls_obj = possible_values[0]

            if isinstance(cls_obj, KnownValue):
                cls_obj = KnownValue(UnwrapClass.unwrap(cls_obj.val, self.options))
                current_class = cls_obj.val
                if isinstance(current_class, type):
                    self._record_class_examined(current_class)
                else:
                    current_class = None
            else:
                current_class = None

            with self.scopes.add_scope(
                ScopeType.class_scope, scope_node=None, scope_object=current_class
            ), self._set_current_class(current_class):
                self._generic_visit_list(node.body)

            if isinstance(cls_obj, KnownValue):
                return cls_obj

        return AnyValue(AnySource.inference)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Value:
        return self.visit_FunctionDef(node)

    def visit_FunctionDef(self, node: FunctionDefNode) -> Value:
        potential_function = self._get_potential_function(node)
        info = compute_function_info(
            node,
            self,
            # If we set the current_class in the collecting phase,
            # the self argument of nested methods with an unannotated
            # first argument is incorrectly inferred.
            enclosing_class=TypedValue(self.current_class)
            if self.current_class is not None and self._is_checking()
            else None,
            is_nested_in_class=self.node_context.includes(ast.ClassDef),
            potential_function=potential_function,
        )

        self.yield_checker.reset_yield_checks()

        if node.returns is None:
            self._show_error_if_checking(
                node, error_code=ErrorCode.missing_return_annotation
            )

        computed_function = compute_value_of_function(info, self)
        if potential_function is None:
            val = computed_function
        else:
            val = KnownValue(potential_function)
        if not info.is_overload and not info.is_evaluated:
            self._set_name_in_scope(node.name, node, val)

        if (
            node.name in METHODS_ALLOWING_NOTIMPLEMENTED
            and info.return_annotation is not None
        ):
            expected_return = info.return_annotation | KnownValue(NotImplemented)
        else:
            expected_return = info.return_annotation

        with self.asynq_checker.set_func_name(
            node.name, async_kind=info.async_kind, is_classmethod=info.is_classmethod
        ), qcore.override(self, "yield_checker", YieldChecker(self)), qcore.override(
            self, "is_async_def", isinstance(node, ast.AsyncFunctionDef)
        ), qcore.override(
            self, "current_function_name", node.name
        ), qcore.override(
            self, "current_function", potential_function
        ), qcore.override(
            self, "expected_return_value", expected_return
        ), qcore.override(
            self, "current_function_info", info
        ):
            result = self._visit_function_body(info)

        if (
            not result.has_return
            and not info.is_overload
            and not info.is_evaluated
            and not info.is_abstractmethod
            and node.returns is not None
            and info.return_annotation != KnownNone
            and not (
                isinstance(info.return_annotation, AnnotatedValue)
                and info.return_annotation.value == KnownNone
            )
        ):
            if info.return_annotation is NO_RETURN_VALUE:
                self._show_error_if_checking(
                    node, error_code=ErrorCode.no_return_may_return
                )
            else:
                self._show_error_if_checking(node, error_code=ErrorCode.missing_return)

        if node.returns is None:
            if (
                result.has_return
                and not info.is_overload
                and not info.is_abstractmethod
            ):
                prepared = prepare_type(result.return_value)
                if should_suggest_type(prepared):
                    detail, metadata = display_suggested_type(prepared)
                    self._show_error_if_checking(
                        node,
                        error_code=ErrorCode.suggested_return_type,
                        detail=detail,
                        extra_metadata=metadata,
                    )

            if info.async_kind == AsyncFunctionKind.normal and _is_asynq_future(
                result.return_value
            ):
                self._show_error_if_checking(
                    node, error_code=ErrorCode.task_needs_yield
                )

        self._set_argspec_to_retval(val, info, result)
        return val

    def _set_argspec_to_retval(
        self, val: Value, info: FunctionInfo, result: FunctionResult
    ) -> None:
        if isinstance(info.node, ast.Lambda) or info.node.returns is not None:
            return
        if info.async_kind == AsyncFunctionKind.async_proxy:
            # Don't attempt to infer the return value of async_proxy functions, since it will be
            # set within the Future returned. Without this, we'll incorrectly infer the return
            # value to be the Future instead of the Future's value.
            return
        if info.node.decorator_list and not (
            len(info.decorators) == 1
            and info.decorators[0][0] in SAFE_DECORATORS_FOR_ARGSPEC_TO_RETVAL
        ):
            return  # With decorators we don't know what it will return
        return_value = result.return_value

        if result.is_generator and return_value == KnownNone:
            return_value = AnyValue(AnySource.inference)

        # pure async functions are otherwise incorrectly inferred as returning whatever the
        # underlying function returns
        if info.async_kind == AsyncFunctionKind.pure:
            task_cls = _get_task_cls(info.potential_function)
            return_value = AsyncTaskIncompleteValue(task_cls, return_value)

        if isinstance(info.node, ast.AsyncFunctionDef) or info.is_decorated_coroutine:
            return_value = make_coro_type(return_value)

        if isinstance(val, KnownValue) and isinstance(val.val, property):
            fget = val.val.fget
            if fget is None:
                return
            val = KnownValue(fget)

        sig = self.signature_from_value(val)
        if sig is None or sig.has_return_value():
            return
        self._argspec_to_retval[id(sig)] = (return_value, sig)

    def _get_potential_function(self, node: FunctionDefNode) -> Optional[object]:
        scope_type = self.scopes.scope_type()
        if scope_type == ScopeType.module_scope and self.module is not None:
            potential_function = safe_getattr(self.module, node.name, None)
        elif scope_type == ScopeType.class_scope and self.current_class is not None:
            potential_function = safe_getattr(self.current_class, node.name, None)
        else:
            potential_function = None

        if (
            potential_function is not None
            and self.options.is_error_code_enabled_anywhere(
                ErrorCode.suggested_parameter_type
            )
        ):
            sig = self.signature_from_value(KnownValue(potential_function))
            if isinstance(sig, Signature):
                self.checker.callable_tracker.record_callable(
                    node, potential_function, sig, self
                )
        return potential_function

    def record_call(self, callable: object, arguments: CallArgs) -> None:
        if self.options.is_error_code_enabled_anywhere(
            ErrorCode.suggested_parameter_type
        ):
            self.checker.callable_tracker.record_call(callable, arguments)

    def visit_Lambda(self, node: ast.Lambda) -> Value:
        with self.asynq_checker.set_func_name("<lambda>"):
            info = compute_function_info(node, self)
            with qcore.override(self, "current_function_info", info):
                result = self._visit_function_body(info)
            return compute_value_of_function(info, self, result=result.return_value)

    def _visit_function_body(self, function_info: FunctionInfo) -> FunctionResult:
        is_collecting = self._is_collecting()
        node = function_info.node

        class_ctx = (
            qcore.empty_context
            if not self.scopes.is_nested_function()
            else qcore.override(self, "current_class", None)
        )
        with class_ctx:
            self._check_method_first_arg(node, function_info=function_info)
        infos = function_info.params
        params = [info.param for info in infos]

        if is_collecting and not self.scopes.contains_scope_of_type(
            ScopeType.function_scope
        ):
            return FunctionResult(parameters=params)

        if function_info.is_evaluated:
            if self._is_collecting() or isinstance(node, ast.Lambda):
                return FunctionResult(parameters=params)
            with self.scopes.allow_only_module_scope():
                # The return annotation doesn't actually matter for validation.
                evaluator = SyntheticEvaluator.from_visitor(
                    node, self, AnyValue(AnySource.marker)
                )
                ctx = type_evaluation.EvalContext(
                    variables={param.name: param.annotation for param in params},
                    positions={param.name: type_evaluation.DEFAULT for param in params},
                    can_assign_context=self,
                    tv_map={},
                )
                for error in evaluator.validate(ctx):
                    self.show_error(
                        error.node, error.message, error_code=ErrorCode.bad_evaluator
                    )
                if self.annotate:
                    with self.catch_errors(), self.scopes.add_scope(
                        ScopeType.function_scope, scope_node=node
                    ):
                        self._generic_visit_list(node.body)
            return FunctionResult(parameters=params)

        # We pass in the node to add_scope() and visit the body once in collecting
        # mode if in a nested function, so that constraints on nonlocals in the outer
        # scope propagate into this scope. This means that we'll use the constraints
        # of the place where the function is defined, not those of where the function
        # is called, which is strictly speaking wrong but should be fine in practice.
        with self.scopes.add_scope(
            ScopeType.function_scope, scope_node=node
        ), qcore.override(self, "is_generator", False), qcore.override(
            self, "async_kind", function_info.async_kind
        ), qcore.override(
            self, "_name_node_to_statement", {}
        ):
            scope = self.scopes.current_scope()
            assert isinstance(scope, FunctionScope)

            for info in infos:
                if info.is_self:
                    # we need this for the implementation of super()
                    self.scopes.set(
                        "%first_arg",
                        info.param.annotation,
                        "%first_arg",
                        VisitorState.check_names,
                    )
                self.scopes.set(
                    info.param.name,
                    info.param.annotation,
                    info.node,
                    VisitorState.check_names,
                )

            with qcore.override(
                self, "state", VisitorState.collect_names
            ), qcore.override(
                self, "return_values", []
            ), self.yield_checker.set_function_node(
                node
            ):
                if isinstance(node, ast.Lambda):
                    self.visit(node.body)
                else:
                    self._generic_visit_list(node.body)
                scope.get_local(LEAVES_SCOPE, node, self.state)
            if is_collecting:
                return FunctionResult(is_generator=self.is_generator, parameters=params)

            # otherwise we may end up using results from the last yield (generated during the
            # collect state) to evaluate the first one visited during the check state
            self.yield_checker.reset_yield_checks()

            with qcore.override(self, "current_class", None), qcore.override(
                self, "state", VisitorState.check_names
            ), qcore.override(
                self, "return_values", []
            ), self.yield_checker.set_function_node(
                node
            ):
                if isinstance(node, ast.Lambda):
                    return_values = [self.visit(node.body)]
                else:
                    self._generic_visit_list(node.body)
                    return_values = self.return_values
                return_set, _ = scope.get_local(LEAVES_SCOPE, node, self.state)

            self._check_function_unused_vars(scope)
            return self._compute_return_type(
                node, return_values, return_set, function_info, params
            )

    def _compute_return_type(
        self,
        node: FunctionNode,
        return_values: Sequence[Optional[Value]],
        return_set: Value,
        info: FunctionInfo,
        params: Sequence[SigParameter],
    ) -> FunctionResult:
        # Ignore generators for now.
        if (
            isinstance(return_set, AnyValue)
            or return_set is NO_RETURN_VALUE
            or (self.is_generator and info.async_kind is not AsyncFunctionKind.normal)
        ):
            has_return = True
        elif return_set is UNINITIALIZED_VALUE:
            has_return = False
        else:
            assert False, return_set
        # if the return value was never set, the function returns None
        if not return_values:
            return FunctionResult(KnownNone, params, has_return, self.is_generator)
        # None is added to return_values if the function raises an error.
        return_values = [val for val in return_values if val is not None]
        # If it only ever raises an error, we don't know what it returns. Strictly
        # this should perhaps be NoReturnValue, but that leads to issues because
        # in practice this condition often occurs in abstract methods that just
        # raise NotImplementedError.
        if not return_values:
            ret = AnyValue(AnySource.inference)
        else:
            ret = unite_values(*return_values)
        if isinstance(node, ast.Lambda):
            has_return_annotation = False
        else:
            has_return_annotation = node.returns is not None
        return FunctionResult(
            ret,
            params,
            has_return=has_return,
            is_generator=self.is_generator,
            has_return_annotation=has_return_annotation,
        )

    def _check_function_unused_vars(
        self, scope: FunctionScope, enclosing_statement: Optional[ast.stmt] = None
    ) -> None:
        """Shows errors for any unused variables in the function."""
        all_def_nodes = set(
            chain.from_iterable(scope.name_to_all_definition_nodes.values())
        )
        all_used_def_nodes = set(
            chain.from_iterable(scope.usage_to_definition_nodes.values())
        )
        all_unused_nodes = all_def_nodes - all_used_def_nodes
        for unused in all_unused_nodes:
            # Ignore names not defined through a Name node (e.g., function arguments)
            if not isinstance(unused, ast.Name) or not self._is_write_ctx(unused.ctx):
                continue
            # Ignore names that are meant to be ignored
            if unused.id.startswith("_"):
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
                statement = self._name_node_to_statement.get(unused)
                if isinstance(statement, ast.Assign):
                    # it's an assignment
                    if not (
                        isinstance(statement.value, ast.Yield)
                        and isinstance(statement.value.value, ast.Tuple)
                    ):
                        # but not an assignment originating from yielding a tuple (which is
                        # probably an async yield)

                        # We need to loop over the targets to handle code like "a, b = c = func()".
                        # If the target containing our unused variable is a tuple and some of its
                        # members are not unused, ignore it.
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
                elif isinstance(statement, ast.AnnAssign):
                    # ignore assignments in AnnAssign nodes, which don't actually
                    # bind the name
                    continue
            if all(
                node in all_unused_nodes
                for node in scope.name_to_all_definition_nodes[unused.id]
            ):
                self._show_error_if_checking(
                    unused,
                    f"Variable {unused.id} is never accessed",
                    error_code=ErrorCode.unused_variable,
                    replacement=replacement,
                )
            else:
                self._show_error_if_checking(
                    unused,
                    f"Assigned value of {unused.id} is never accessed",
                    error_code=ErrorCode.unused_assignment,
                    replacement=replacement,
                )

    def value_of_annotation(
        self, node: ast.expr, *, allow_unpack: bool = False
    ) -> Value:
        with qcore.override(self, "state", VisitorState.collect_names):
            annotated_type = self._visit_annotation(node)
        return self._value_of_annotation_type(
            annotated_type, node, allow_unpack=allow_unpack
        )

    def _visit_annotation(self, node: ast.AST) -> Value:
        with qcore.override(self, "in_annotation", True):
            return self.visit(node)

    def _value_of_annotation_type(
        self,
        val: Value,
        node: ast.AST,
        *,
        is_typeddict: bool = False,
        allow_unpack: bool = False,
    ) -> Value:
        """Given a value encountered in a type annotation, return a type."""
        return type_from_value(
            val,
            visitor=self,
            node=node,
            is_typeddict=is_typeddict,
            allow_unpack=allow_unpack,
        )

    def _check_method_first_arg(
        self, node: FunctionNode, function_info: FunctionInfo
    ) -> None:
        """Makes sure the first argument to a method is self or cls."""
        if self.current_class is None:
            return
        # staticmethods have no restrictions
        if function_info.is_staticmethod:
            return
        # try to confirm that it's actually a method
        if isinstance(node, ast.Lambda) or not hasattr(self.current_class, node.name):
            return
        if node.name in IMPLICIT_CLASSMETHODS:
            return
        first_must_be = "cls" if function_info.is_classmethod else "self"

        if len(node.args.args) < 1 or len(node.args.defaults) == len(node.args.args):
            self.show_error(
                node,
                "Method must have at least one non-keyword argument",
                ErrorCode.method_first_arg,
            )
        elif node.args.args[0].arg != first_must_be:
            self.show_error(
                node,
                f"First argument to method should be {first_must_be}",
                ErrorCode.method_first_arg,
            )

    def visit_Global(self, node: ast.Global) -> None:
        if self.scopes.scope_type() != ScopeType.function_scope:
            self._show_error_if_checking(node, error_code=ErrorCode.bad_global)
            return

        module_scope = self.scopes.module_scope()
        for name in node.names:
            if self.unused_finder is not None and module_scope.scope_object is not None:
                assert isinstance(module_scope.scope_object, types.ModuleType)
                self.unused_finder.record(
                    module_scope.scope_object, name, module_scope.scope_object.__name__
                )
            self._set_name_in_scope(name, node, ReferencingValue(module_scope, name))

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        if self.scopes.scope_type() != ScopeType.function_scope:
            self._show_error_if_checking(node, error_code=ErrorCode.bad_nonlocal)
            return

        for name in node.names:
            defining_scope = self.scopes.get_nonlocal_scope(
                name, self.scopes.current_scope()
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
                defining_scope = self.scopes.module_scope()
            self._set_name_in_scope(name, node, ReferencingValue(defining_scope, name))

    def check_deprecation(self, node: ast.AST, value: Value) -> bool:
        if isinstance(value, AnnotatedValue):
            if value.has_metadata_of_type(SkipDeprecatedExtension):
                return False
            for metadata in value.get_metadata_of_type(DeprecatedExtension):
                self._show_error_if_checking(
                    node,
                    f"{value} is deprecated: {metadata.deprecation_message}",
                    error_code=ErrorCode.deprecated,
                )
                return True
            return self.check_deprecation(node, value.value)
        if isinstance(value, UnboundMethodValue):
            method = value.get_method()
            if method is None:
                return False
            return self.check_deprecation(node, KnownValue(method))
        if isinstance(value, CallableValue):
            if not isinstance(value.signature, Signature):
                return False
            if value.signature.deprecated is None:
                return False
            deprecated = value.signature.deprecated
        elif isinstance(value, KnownValue):
            deprecated = safe_getattr(value.val, "__deprecated__", None)
            if deprecated is None:
                return False
        else:
            return False
        if not safe_isinstance(deprecated, str):
            # happens with Mock objects
            return False
        self._show_error_if_checking(
            node,
            f"{value} is deprecated: {deprecated}",
            error_code=ErrorCode.deprecated,
        )
        return True

    # Imports

    def visit_Import(self, node: ast.Import) -> None:
        self.generic_visit(node)
        if self.scopes.scope_type() == ScopeType.module_scope:
            for name in node.names:
                self.import_name_to_node[name.name] = node

        for alias in node.names:
            self._try_to_import(alias.name)
            # "import a.b" sets the name "a", but "import a.b as c" sets "c" to the value "a.b"
            varname = (
                alias.name if alias.asname is not None else alias.name.split(".")[0]
            )
            mod = self._get_module(varname, node)
            self._set_alias_in_scope(alias, mod, node=node)

    def _set_alias_in_scope(
        self,
        alias: ast.alias,
        value: Value,
        *,
        force_public: bool = False,
        node: ast.AST,
    ) -> None:
        # aliases only have a lineno attached in 3.10+
        if sys.version_info >= (3, 10):
            error_node = alias
        else:
            error_node = node
        if self.check_deprecation(error_node, value):
            value = annotate_value(value, [SkipDeprecatedExtension()])
        if alias.asname is not None:
            self._set_name_in_scope(
                alias.asname,
                alias,
                value,
                private=not force_public and alias.asname != alias.name,
            )
        else:
            self._set_name_in_scope(
                alias.name.split(".")[0], alias, value, private=not force_public
            )

    def _get_module(self, name: str, node: ast.AST) -> Value:
        if name not in sys.modules:
            self._try_to_import(name)
        if name in sys.modules:
            # import a.b.c only succeeds if a.b.c is a module that
            # exists, but it doesn't return the module a.b.c, it
            # follows the attribute chain. But this isn't true for
            # ImportFrom.
            if isinstance(node, ast.ImportFrom):
                return KnownValue(sys.modules[name])
            pieces = name.split(".")
            base_module = sys.modules.get(pieces[0])
            for piece in pieces[1:]:
                if not safe_hasattr(base_module, piece):
                    self._show_error_if_checking(
                        node,
                        (
                            f"Cannot import {name} because {piece} is not an attribute"
                            f" of {base_module!r}"
                        ),
                        error_code=ErrorCode.import_failed,
                    )
                    return AnyValue(AnySource.unresolved_import)
                base_module = getattr(base_module, piece)
            return KnownValue(base_module)
        else:
            # TODO: Maybe get the module from stubs?
            self._show_error_if_checking(
                node, f"Cannot import {name}", error_code=ErrorCode.import_failed
            )
            return AnyValue(AnySource.unresolved_import)

    def _try_to_import(self, module_name: str) -> None:
        try:
            __import__(module_name)
        except Exception:
            pass

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.generic_visit(node)
        # this is used to decide where to add additional imports (after the first import), so
        # exclude __future__ imports
        if (
            self.scopes.scope_type() == ScopeType.module_scope
            and node.module
            and node.module != "__future__"
        ):
            self.import_name_to_node[node.module] = node
        if node.module == "__future__":
            for name in node.names:
                self.future_imports.add(name.name)

        self._maybe_record_usages_from_import(node)

        # See if we can get the names from the stub instead
        if (
            node.module is not None
            and node.level == 0
            # pyanalyze.extensions has a stub only for the purpose of other stubs
            # it shouldn't be used for runtime imports
            and node.module != "pyanalyze.extensions"
        ):
            path = typeshed_client.ModulePath(tuple(node.module.split(".")))
            finder = self.checker.ts_finder
            mod = finder.resolver.get_module(path)
            if mod.exists:
                for alias in node.names:
                    val = finder.resolve_name(node.module, alias.name)
                    if val is UNINITIALIZED_VALUE:
                        self._show_error_if_checking(
                            node,
                            f"Cannot import name {alias.name!r} from {node.module!r}",
                            ErrorCode.import_failed,
                        )
                        val = AnyValue(AnySource.error)
                    self._set_alias_in_scope(alias, val, node=node)
                return

        is_init = self.filename.endswith("/__init__.py")
        source_module = self._get_import_from_module(node)

        # from .a import b implicitly sets a in the parent module's namespace.
        # We allow relying on this behavior.
        if (
            is_init
            and node.module is not None
            and "." not in node.module
            and node.level == 1
        ):
            self._set_name_in_scope(node.module, node, source_module, private=False)

        for alias in node.names:
            if alias.name == "*":
                if isinstance(source_module, KnownValue) and isinstance(
                    source_module.val, types.ModuleType
                ):
                    for name, val in source_module.val.__dict__.items():
                        if name.startswith("_"):
                            continue
                        self._set_name_in_scope(
                            name, alias, KnownValue(val), private=False
                        )
                else:
                    self._show_error_if_checking(
                        node,
                        f"Cannot import * from unresolved module {node.module!r}",
                        ErrorCode.invalid_import,
                    )
                continue
            val = self._get_import_from_value(source_module, alias.name, node)
            self._set_alias_in_scope(
                alias, val, force_public=is_init and node.level == 1, node=node
            )

    def _get_import_from_value(
        self, source_module: Value, alias_name: str, node: ast.ImportFrom
    ) -> Value:
        val = self.get_attribute_from_value(source_module, alias_name)
        if val is not UNINITIALIZED_VALUE:
            return val
        if isinstance(source_module, KnownValue) and isinstance(
            source_module.val, types.ModuleType
        ):
            name = f"{source_module.val.__name__}.{alias_name}"
            self._try_to_import(name)
            val = self.get_attribute_from_value(source_module, alias_name)
            if val is not UNINITIALIZED_VALUE:
                return val

        self._show_error_if_checking(
            node,
            f"Cannot import name {alias_name!r} from {node.module!r}",
            ErrorCode.import_failed,
        )
        return AnyValue(AnySource.error)

    def _get_import_from_module(self, node: ast.ImportFrom) -> Value:
        if node.level > 0:
            if self.module is None:
                return AnyValue(AnySource.unresolved_import)
            level = node.level
            if self.filename.endswith("/__init__.py"):
                level -= 1

            current_module_path: List[str] = self.module.__name__.split(".")
            if level >= len(current_module_path):
                self._show_error_if_checking(
                    node,
                    "Attempted relative import beyond top-level package",
                    error_code=ErrorCode.invalid_import,
                )
                return AnyValue(AnySource.error)
            if level:
                current_module_path = current_module_path[:-level]
            if node.module is not None:
                current_module_path.append(node.module)
            module_name = ".".join(current_module_path)
        else:
            # Should be disallowed by the AST
            if node.module is None:
                self._show_error_if_checking(
                    node,
                    "Attempted absolute import without module name",
                    error_code=ErrorCode.invalid_import,
                )
                return AnyValue(AnySource.error)
            module_name = node.module
        return self._get_module(module_name, node)

    def _maybe_record_usages_from_import(self, node: ast.ImportFrom) -> None:
        if self.unused_finder is None or self.module is None:
            return
        if self._is_unimportable_module(node):
            return
        if node.level == 0:
            module_name = node.module
        else:
            if self.filename.endswith("/__init__.py"):
                this_module_name = self.module.__name__ + ".__init__"
            else:
                this_module_name = self.module.__name__
            parent_module_name = this_module_name.rsplit(".", maxsplit=node.level)[0]
            if node.module is not None:
                module_name = parent_module_name + "." + node.module
            else:
                module_name = parent_module_name
        if module_name is None:
            return
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = __import__(module_name)
            except Exception:
                return
        for alias in node.names:
            if alias.name == "*":
                self.unused_finder.record_import_star(module, self.module)
            else:
                self.unused_finder.record(module, alias.name, self.module.__name__)

    def _is_unimportable_module(self, node: Union[ast.Import, ast.ImportFrom]) -> bool:
        unimportable = self.options.get_value_for(UnimportableModules)
        if isinstance(node, ast.ImportFrom):
            # the split is needed for cases like "from foo.bar import baz" if foo is unimportable
            return node.module is not None and node.module.split(".")[0] in unimportable
        else:
            # need the split if the code is "import foo.bar as bar" if foo is unimportable
            return any(name.name.split(".")[0] in unimportable for name in node.names)

    # Comprehensions

    def visit_DictComp(self, node: ast.DictComp) -> Value:
        return self._visit_sequence_comp(node, dict)

    def visit_ListComp(self, node: ast.ListComp) -> Value:
        return self._visit_sequence_comp(node, list)

    def visit_SetComp(self, node: ast.SetComp) -> Value:
        return self._visit_sequence_comp(node, set)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Value:
        return self._visit_sequence_comp(node, types.GeneratorType)

    def visit_comprehension(
        self, node: ast.comprehension, iterable_type: Optional[Value] = None
    ) -> None:
        if iterable_type is None:
            is_async = bool(node.is_async)
            iterable_type = self._member_value_of_iterator(node.iter, is_async)
            if not isinstance(iterable_type, Value):
                iterable_type = unite_and_simplify(
                    *iterable_type,
                    limit=self.options.get_value_for(UnionSimplificationLimit),
                )
        with qcore.override(self, "in_comprehension_body", True):
            with qcore.override(self, "being_assigned", iterable_type):
                self.visit(node.target)
            for cond in node.ifs:
                _, constraint = self.constraint_from_condition(cond)
                self.add_constraint(cond, constraint)

    def _visit_sequence_comp(
        self,
        node: Union[ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp],
        typ: type,
    ) -> Value:
        # the iteree of the first generator is executed in the enclosing scope
        is_async = bool(node.generators[0].is_async)
        iterable_type = self._member_value_of_iterator(
            node.generators[0].iter, is_async
        )
        if self.state == VisitorState.collect_names:
            # Visit it once to get usage nodes for usage of nested variables. This enables
            # us to inherit constraints on nested variables.
            # Strictly speaking this is unsafe to do for generator expressions, which may
            # be evaluated at a different place in the function than where they are defined,
            # but that is unlikely to be an issue in practice.
            with self.scopes.add_scope(
                ScopeType.function_scope, scope_node=node
            ), qcore.override(self, "_name_node_to_statement", {}):
                return self._visit_comprehension_inner(node, typ, iterable_type)

        with self.scopes.add_scope(
            ScopeType.function_scope, scope_node=node
        ), qcore.override(self, "_name_node_to_statement", {}):
            scope = self.scopes.current_scope()
            assert isinstance(scope, FunctionScope)
            for state in (VisitorState.collect_names, VisitorState.check_names):
                with qcore.override(self, "state", state):
                    ret = self._visit_comprehension_inner(node, typ, iterable_type)
            stmt = self.node_context.nearest_enclosing(ast.stmt)
            assert isinstance(stmt, ast.stmt)
            self._check_function_unused_vars(scope, enclosing_statement=stmt)
        return ret

    def _visit_comprehension_inner(
        self,
        node: Union[ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp],
        typ: type,
        iterable_type: Union[Value, Sequence[Value]],
    ) -> Value:
        if not isinstance(iterable_type, Value):
            # If it is a simple comprehension (only one generator, no ifs) and we know
            # the exact iterated values, we try to infer an IncompleteValue instead.
            if (
                len(node.generators) == 1
                and not node.generators[0].ifs
                and 0
                < len(iterable_type)
                <= self.options.get_value_for(ComprehensionLengthInferenceLimit)
            ):
                generator = node.generators[0]
                if isinstance(node, ast.DictComp):
                    items = []
                    self.node_context.contexts.append(generator)
                    try:
                        for val in iterable_type:
                            self.visit_comprehension(generator, iterable_type=val)
                            with qcore.override(self, "in_comprehension_body", True):
                                # PEP 572 mandates that the key be evaluated first.
                                key = self.visit(node.key)
                                value = self.visit(node.value)
                                items.append(KVPair(key, value))
                    finally:
                        self.node_context.contexts.pop()
                    return DictIncompleteValue(typ, items)
                elif isinstance(node, (ast.ListComp, ast.SetComp)):
                    elts = []
                    self.node_context.contexts.append(generator)
                    try:
                        for val in iterable_type:
                            self.visit_comprehension(generator, iterable_type=val)
                            with qcore.override(self, "in_comprehension_body", True):
                                elts.append((False, self.visit(node.elt)))
                    finally:
                        self.node_context.contexts.pop()
                    return SequenceValue(typ, elts)

            iterable_type = unite_and_simplify(
                *iterable_type,
                limit=self.options.get_value_for(UnionSimplificationLimit),
            )
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

        if isinstance(node, ast.DictComp):
            with qcore.override(self, "in_comprehension_body", True):
                key_value = self.visit(node.key)
                value_value = self.visit(node.value)

                hashability = check_hashability(key_value, self)
                if isinstance(hashability, CanAssignError):
                    self._show_error_if_checking(
                        node.key,
                        "Dictionary key is not hashable",
                        ErrorCode.unhashable_key,
                        detail=str(hashability),
                    )
                    key_value = AnyValue(AnySource.error)
            return DictIncompleteValue(
                dict, [KVPair(key_value, value_value, is_many=True)]
            )

        with qcore.override(self, "in_comprehension_body", True):
            member_value = self.visit(node.elt)

            if typ is set:
                hashability = check_hashability(member_value, self)
                if isinstance(hashability, CanAssignError):
                    self._show_error_if_checking(
                        node.elt,
                        "Set member is not hashable",
                        ErrorCode.unhashable_key,
                        detail=str(hashability),
                    )
                    member_value = AnyValue(AnySource.error)

        if typ is types.GeneratorType:
            return GenericValue(typ, [member_value, KnownValue(None), KnownValue(None)])
        # Returning a SequenceValue here instead of a GenericValue allows
        # later code to modify this container.
        return SequenceValue(typ, [(True, member_value)])

    # Literals and displays

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Value:
        elements = self._generic_visit_list(node.values)
        limit = self.options.get_value_for(UnionSimplificationLimit)
        possible_values: List[List[str]] = [[]]
        for elt in elements:
            subvals = list(flatten_values(elt))
            # Bail out if the list of possible values gets too long.
            if len(possible_values) * len(subvals) > limit:
                return TypedValue(str)
            to_add = []
            for subval in subvals:
                if not isinstance(subval, KnownValue):
                    return TypedValue(str)
                if not isinstance(subval.val, str):
                    return TypedValue(str)
                to_add.append(subval.val)
            possible_values = [
                lst + [new_elt] for lst in possible_values for new_elt in to_add
            ]
        return unite_values(*[KnownValue("".join(lst)) for lst in possible_values])

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Value:
        val = self.visit(node.value)
        format_spec_val = (
            self.visit(node.format_spec) if node.format_spec else KnownValue("")
        )
        if isinstance(format_spec_val, KnownValue) and isinstance(
            format_spec_val.val, str
        ):
            format_spec = format_spec_val.val
        else:
            # TODO: statically check whether the format specifier is valid.
            return TypedValue(str)
        possible_vals = []
        for subval in flatten_values(val):
            possible_vals.append(
                self._visit_single_formatted_value(subval, node, format_spec)
            )
        return unite_and_simplify(
            *possible_vals, limit=self.options.get_value_for(UnionSimplificationLimit)
        )

    def _visit_single_formatted_value(
        self, val: Value, node: ast.FormattedValue, format_spec: str
    ) -> Value:
        if not isinstance(val, KnownValue):
            return TypedValue(str)
        output = val.val
        if node.conversion != -1:
            unsupported_conversion = False
            try:
                if node.conversion == ord("a"):
                    output = ascii(output)
                elif node.conversion == ord("s"):
                    output = str(output)
                elif node.conversion == ord("r"):
                    output = repr(output)
                else:
                    unsupported_conversion = True
            except Exception:
                # str/repr/ascii failed
                return TypedValue(str)
            if unsupported_conversion:
                raise NotImplementedError(
                    f"Unsupported converion specifier {node.conversion}"
                )
        try:
            output = format(output, format_spec)
        except Exception:
            # format failed
            return TypedValue(str)
        return KnownValue(output)

    def visit_Constant(self, node: ast.Constant) -> Value:
        # replaces Num, Str, etc. in 3.8+
        if isinstance(node.value, str):
            self._maybe_show_missing_f_error(node, node.value)
        return KnownValue(node.value)

    def visit_Num(self, node: ast.Num) -> Value:
        return KnownValue(node.n)

    def visit_Str(self, node: ast.Str) -> Value:
        self._maybe_show_missing_f_error(node, node.s)
        return KnownValue(node.s)

    def _maybe_show_missing_f_error(self, node: ast.AST, s: Union[str, bytes]) -> None:
        """Show an error if this string was probably meant to be an f-string."""
        if isinstance(s, bytes):
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
            stmt = f_str_ast.body[0]
            assert isinstance(stmt, ast.Expr), f"unexpected ast {ast.dump(f_str_ast)}"
            self._show_error_if_checking(
                node,
                error_code=ErrorCode.missing_f,
                replacement=self.replace_node(node, stmt.value),
            )

    def _name_exists(self, name: str) -> bool:
        try:
            val = self.scopes.get(name, None, VisitorState.check_names)
        except KeyError:
            return False
        else:
            return val is not UNINITIALIZED_VALUE

    def visit_Bytes(self, node: ast.Bytes) -> Value:
        return KnownValue(node.s)

    def visit_NameConstant(self, node: ast.NameConstant) -> Value:
        # True, False, None in py3
        return KnownValue(node.value)

    def visit_Dict(self, node: ast.Dict) -> Value:
        ret = {}
        all_pairs: List[KVPair] = []
        has_non_literal = False
        for key_node, value_node in zip(node.keys, node.values):
            value_val = self.visit(value_node)
            # ** unpacking
            if key_node is None:
                has_non_literal = True
                new_pairs = kv_pairs_from_mapping(value_val, self)
                if isinstance(new_pairs, CanAssignError):
                    self._show_error_if_checking(
                        value_node,
                        f"{value_val} is not a mapping",
                        ErrorCode.unsupported_operation,
                        detail=str(new_pairs),
                    )
                    return TypedValue(dict)
                all_pairs += new_pairs
                continue
            key_val = self.visit(key_node)

            hashability = check_hashability(key_val, self)
            if isinstance(hashability, CanAssignError):
                self._show_error_if_checking(
                    key_node,
                    "Dictionary key is not hashable",
                    ErrorCode.unhashable_key,
                    detail=str(hashability),
                )

            all_pairs.append(KVPair(key_val, value_val))
            if not isinstance(key_val, KnownValue) or not isinstance(
                value_val, KnownValue
            ):
                has_non_literal = True
            value = value_val.val if isinstance(value_val, KnownValue) else None

            if not isinstance(key_val, KnownValue):
                continue

            key = key_val.val

            try:
                already_exists = key in ret
            except TypeError as e:
                continue

            if already_exists:
                self._show_error_if_checking(
                    key_node,
                    f"Duplicate dictionary key {key!r}",
                    ErrorCode.duplicate_dict_key,
                )
            ret[key] = value

        if has_non_literal:
            return DictIncompleteValue(dict, all_pairs)
        else:
            return KnownValue(ret)

    def visit_Set(self, node: ast.Set) -> Value:
        return self._visit_display_read(node, set)

    def visit_List(self, node: ast.List) -> Optional[Value]:
        return self._visit_display(node, list)

    def visit_Tuple(self, node: ast.Tuple) -> Optional[Value]:
        return self._visit_display(node, tuple)

    def _visit_display(
        self, node: Union[ast.List, ast.Tuple], typ: type
    ) -> Optional[Value]:
        if self._is_write_ctx(node.ctx):
            target_length = 0
            post_starred_length = None
            for target in node.elts:
                if isinstance(target, ast.Starred):
                    if post_starred_length is not None:
                        # This is a SyntaxError at runtime so it should never happen
                        self.show_error(
                            node,
                            "Two starred expressions in assignment",
                            error_code=ErrorCode.unexpected_node,
                        )
                        with qcore.override(
                            self, "being_assigned", AnyValue(AnySource.error)
                        ):
                            return self.generic_visit(node)
                    else:
                        post_starred_length = 0
                elif post_starred_length is not None:
                    post_starred_length += 1
                else:
                    target_length += 1

            assert (
                self.being_assigned is not None
            ), "annotated assignment can only have a single target"
            being_assigned = unpack_values(
                self.being_assigned, self, target_length, post_starred_length
            )
            if isinstance(being_assigned, CanAssignError):
                self.show_error(
                    node,
                    f"Cannot unpack {self.being_assigned}",
                    ErrorCode.bad_unpack,
                    detail=str(being_assigned),
                )
                with qcore.override(self, "being_assigned", AnyValue(AnySource.error)):
                    return self.generic_visit(node)

            for target, value in zip(node.elts, being_assigned):
                with qcore.override(self, "being_assigned", value):
                    self.visit(target)
            return None
        else:
            return self._visit_display_read(node, typ)

    def _visit_display_read(
        self, node: Union[ast.Set, ast.List, ast.Tuple], typ: type
    ) -> Value:
        elts = [self.visit(elt) for elt in node.elts]
        return self._maybe_make_sequence(typ, elts, node, elt_nodes=node.elts)

    def _maybe_make_sequence(
        self,
        typ: type,
        elts: Sequence[Value],
        node: ast.AST,
        elt_nodes: Optional[Sequence[ast.AST]] = None,
    ) -> Value:
        values = []
        for i, elt in enumerate(elts):
            if isinstance(elt, _StarredValue):
                vals = concrete_values_from_iterable(elt.value, self)
                if isinstance(vals, CanAssignError):
                    self.show_error(
                        elt.node,
                        f"{elt.value} is not iterable",
                        ErrorCode.unsupported_operation,
                        detail=str(vals),
                    )
                    new_vals = [(True, AnyValue(AnySource.error))]
                elif isinstance(vals, Value):
                    # single value
                    new_vals = [(True, vals)]
                else:
                    new_vals = [(False, val) for val in vals]
                if typ is set:
                    for _, val in new_vals:
                        hashability = check_hashability(val, self)
                        if isinstance(hashability, CanAssignError):
                            if elt_nodes:
                                error_node = elt_nodes[i]
                            else:
                                error_node = node
                            self._show_error_if_checking(
                                error_node,
                                "Set element is not hashable",
                                ErrorCode.unhashable_key,
                                detail=str(hashability),
                            )

                values += new_vals
            else:
                if typ is set:
                    hashability = check_hashability(elt, self)
                    if isinstance(hashability, CanAssignError):
                        if elt_nodes:
                            error_node = elt_nodes[i]
                        else:
                            error_node = node
                        self._show_error_if_checking(
                            error_node,
                            "Set element is not hashable",
                            ErrorCode.unhashable_key,
                            detail=str(hashability),
                        )
                values.append((False, elt))

        return SequenceValue.make_or_known(typ, values)

    # Operations

    def visit_BoolOp(self, node: ast.BoolOp) -> Value:
        # Visit an AND or OR expression.

        # We want to show an error if the left operand in a BoolOp is always true,
        # so we use constraint_from_condition.

        # Within the BoolOp itself we set additional constraints: for an AND
        # clause we know that if it is executed, all constraints to its left must
        # be true, so we set a positive constraint; for OR it is the opposite, so
        # we set a negative constraint.

        is_and = isinstance(node.op, ast.And)
        stack = contextlib.ExitStack()
        scopes = []
        out_constraints = []
        values = []
        constraint = NULL_CONSTRAINT
        with stack:
            for i, condition in enumerate(node.values):
                is_last = i == len(node.values) - 1
                scope = stack.enter_context(self.scopes.subscope())
                scopes.append(scope)
                if is_and:
                    self.add_constraint(condition, constraint)
                else:
                    self.add_constraint(condition, constraint.invert())

                new_value, constraint = self.constraint_from_condition(
                    condition, check_boolability=not is_last
                )
                out_constraints.append(constraint)

                if is_last:
                    values.append(new_value)
                elif is_and:
                    values.append(constrain_value(new_value, FALSY_CONSTRAINT))
                else:
                    values.append(constrain_value(new_value, TRUTHY_CONSTRAINT))

        self.scopes.combine_subscopes(scopes)
        constraint_cls = AndConstraint if is_and else OrConstraint
        constraint = constraint_cls.make(reversed(out_constraints))
        return annotate_with_constraint(unite_values(*values), constraint)

    def visit_Compare(self, node: ast.Compare) -> Value:
        nodes = [node.left, *node.comparators]
        vals = [self._visit_possible_constraint(node) for node in nodes]
        results = []
        constraints = []
        for i, (rhs_node, rhs) in enumerate(zip(nodes, vals)):
            if i == 0:
                continue
            op = node.ops[i - 1]
            lhs_node = nodes[i - 1]
            lhs = vals[i - 1]
            result = self._visit_single_compare(lhs_node, lhs, op, rhs_node, rhs, node)
            constraints.append(extract_constraints(result))
            result, _ = unannotate_value(result, ConstraintExtension)
            results.append(result)
        return annotate_with_constraint(
            unite_values(*results), AndConstraint.make(constraints)
        )

    def _visit_single_compare(
        self,
        lhs_node: ast.AST,
        lhs: Value,
        op: ast.AST,
        rhs_node: ast.AST,
        rhs: Value,
        parent_node: ast.AST,
    ) -> Value:
        lhs_constraint = extract_constraints(lhs)
        rhs_constraint = extract_constraints(rhs)
        if isinstance(lhs, AnnotatedValue):
            lhs = lhs.value
        if isinstance(rhs, AnnotatedValue):
            rhs = rhs.value
        if isinstance(lhs_constraint, PredicateProvider) and isinstance(
            rhs, KnownValue
        ):
            constraint = self._constraint_from_predicate_provider(
                lhs_constraint, rhs.val, op
            )
        elif isinstance(rhs_constraint, PredicateProvider) and isinstance(
            lhs, KnownValue
        ):
            constraint = self._constraint_from_predicate_provider(
                rhs_constraint, lhs.val, op
            )
        elif isinstance(rhs, KnownValue):
            constraint = self._constraint_from_compare_op(
                lhs_node, rhs.val, op, is_right=True
            )
        elif isinstance(lhs, KnownValue):
            constraint = self._constraint_from_compare_op(
                rhs_node, lhs.val, op, is_right=False
            )
        else:
            constraint = NULL_CONSTRAINT
        if isinstance(op, (ast.Is, ast.IsNot)):
            # is and is not always return a boolean and don't forward to a dunder.
            val = TypedValue(bool)
        elif isinstance(op, (ast.In, ast.NotIn)):
            self._visit_binop_internal(
                rhs_node,
                Composite(rhs),
                op,
                lhs_node,
                Composite(lhs),
                parent_node,
                allow_call=False,
            )
            # These always return a bool, regardless of what the dunder does.
            val = TypedValue(bool)
        else:
            val = self._visit_binop_internal(
                lhs_node,
                Composite(lhs),
                op,
                rhs_node,
                Composite(rhs),
                parent_node,
                allow_call=False,
            )

        return annotate_with_constraint(val, constraint)

    def _constraint_from_compare_op(
        self, constrained_node: ast.AST, other_val: Any, op: ast.AST, *, is_right: bool
    ) -> AbstractConstraint:
        varname = self.composite_from_node(constrained_node).varname
        if varname is None:
            return NULL_CONSTRAINT
        if isinstance(op, (ast.Is, ast.IsNot)):
            predicate = EqualsPredicate(other_val, self, use_is=True)
            positive = isinstance(op, ast.Is)
            return Constraint(varname, ConstraintType.predicate, positive, predicate)
        elif isinstance(op, (ast.Eq, ast.NotEq)):
            predicate = EqualsPredicate(other_val, self)
            positive = isinstance(op, ast.Eq)
            return Constraint(varname, ConstraintType.predicate, positive, predicate)
        elif isinstance(op, (ast.In, ast.NotIn)) and is_right:

            def predicate_func(value: Value, positive: bool) -> Optional[Value]:
                op = _in if positive else _not_in
                if isinstance(value, KnownValue):
                    try:
                        result = op(value.val, other_val)
                    except Exception:
                        pass
                    else:
                        if not result:
                            return None
                elif positive:
                    known_other = KnownValue(other_val)
                    member_values = concrete_values_from_iterable(known_other, self)
                    if isinstance(member_values, CanAssignError):
                        return value
                    elif isinstance(member_values, Value):
                        if value.is_assignable(member_values, self):
                            return member_values
                        return None
                    else:
                        possible_values = [
                            val
                            for val in member_values
                            if value.is_assignable(val, self)
                        ]
                        return unite_values(*possible_values)
                return value

            positive = isinstance(op, ast.In)
            return Constraint(
                varname, ConstraintType.predicate, positive, predicate_func
            )
        else:
            positive_operator, negative_operator = COMPARATOR_TO_OPERATOR[type(op)]

            def predicate_func(value: Value, positive: bool) -> Optional[Value]:
                op = positive_operator if positive else negative_operator
                if isinstance(value, KnownValue):
                    try:
                        if is_right:
                            result = op(value.val, other_val)
                        else:
                            result = op(other_val, value.val)
                    except Exception:
                        pass
                    else:
                        if not result:
                            return None
                return value

            return Constraint(varname, ConstraintType.predicate, True, predicate_func)

    def _constraint_from_predicate_provider(
        self, pred: PredicateProvider, other_val: Any, op: ast.AST
    ) -> Constraint:
        positive_operator, negative_operator = COMPARATOR_TO_OPERATOR[type(op)]

        def predicate_func(value: Value, positive: bool) -> Optional[Value]:
            predicate_value = pred.provider(value)
            if isinstance(predicate_value, KnownValue):
                op = positive_operator if positive else negative_operator
                try:
                    result = op(predicate_value.val, other_val)
                except Exception:
                    pass
                else:
                    if not result:
                        return None
            return value

        return Constraint(pred.varname, ConstraintType.predicate, True, predicate_func)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Value:
        if isinstance(node.op, ast.Not):
            # not doesn't have its own special method
            val, constraint = self.constraint_from_condition(node.operand)
            boolability = get_boolability(val)
            if boolability.is_safely_true():
                val = KnownValue(False)
            elif boolability.is_safely_false():
                val = KnownValue(True)
            else:
                val = TypedValue(bool)
            return annotate_with_constraint(val, constraint.invert())
        else:
            operand = self.composite_from_node(node.operand)
            _, method = UNARY_OPERATION_TO_DESCRIPTION_AND_METHOD[type(node.op)]
            val, _ = self._check_dunder_call(node, operand, method, [], allow_call=True)
            return val

    def visit_BinOp(self, node: ast.BinOp) -> Value:
        left = self.composite_from_node(node.left)
        right = self.composite_from_node(node.right)
        return self._visit_binop_internal(
            node.left, left, node.op, node.right, right, node
        )

    def _visit_binop_internal(
        self,
        left_node: ast.AST,
        left_composite: Composite,
        op: ast.AST,
        right_node: ast.AST,
        right_composite: Composite,
        source_node: ast.AST,
        is_inplace: bool = False,
        allow_call: bool = True,
    ) -> Value:
        left = left_composite.value
        right = right_composite.value
        if self.in_annotation and isinstance(op, ast.BitOr):
            # Accept PEP 604 (int | None) in annotations
            if isinstance(left, KnownValue) and isinstance(right, KnownValue):
                return KnownValue(Union[left.val, right.val])
            else:
                self._show_error_if_checking(
                    source_node,
                    f"Unsupported operands for | in annotation: {left} and {right}",
                    error_code=ErrorCode.unsupported_operation,
                )
                return AnyValue(AnySource.error)

        if (
            isinstance(op, ast.Mod)
            and isinstance(left, KnownValue)
            and isinstance(left.val, (bytes, str))
        ):
            value, replacement_node = format_strings.check_string_format(
                left_node,
                left.val,
                right_node,
                right,
                self._show_error_if_checking,
                self,
            )
            if replacement_node is not None and isinstance(source_node, ast.BinOp):
                replacement = self.replace_node(source_node, replacement_node)
                self._show_error_if_checking(
                    source_node,
                    error_code=ErrorCode.use_fstrings,
                    replacement=replacement,
                )
            return value

        (_, method, imethod, _) = BINARY_OPERATION_TO_DESCRIPTION_AND_METHOD[type(op)]
        allow_call = allow_call and method not in self.options.get_value_for(
            DisallowCallsToDunders
        )

        if is_inplace:
            assert imethod is not None, f"no inplace method available for {op}"
            with self.catch_errors() as inplace_errors:
                # Not _check_dunder_call_or_catch because if the call doesn't
                # typecheck it normally returns NotImplemented and we try the
                # non-inplace method next.
                inplace_result, _ = self._check_dunder_call(
                    source_node,
                    left_composite,
                    imethod,
                    [right_composite],
                    allow_call=allow_call,
                )
            if not inplace_errors:
                return inplace_result

        possibilities = []
        for subval in flatten_values(left):
            result = self._visit_binop_no_mvv(
                Composite(subval, left_composite.varname, left_composite.node),
                op,
                right_composite,
                source_node,
                allow_call,
            )
            possibilities.append(result)
        return unite_values(*possibilities)

    def _visit_binop_no_mvv(
        self,
        left_composite: Composite,
        op: ast.AST,
        right_composite: Composite,
        source_node: ast.AST,
        allow_call: bool = True,
    ) -> Value:
        left = left_composite.value
        right = right_composite.value
        (description, method, _, rmethod) = BINARY_OPERATION_TO_DESCRIPTION_AND_METHOD[
            type(op)
        ]
        if rmethod is None:
            # "in" falls back to __iter__ and then to __getitem__ if __contains__ is not defined
            if method == "__contains__":
                contains_result_or_errors = self._check_dunder_call_or_catch(
                    source_node,
                    left_composite,
                    method,
                    [right_composite],
                    allow_call=allow_call,
                )
                if isinstance(contains_result_or_errors, Value):
                    return contains_result_or_errors

                iterable_type = is_iterable(left, self)
                if isinstance(iterable_type, Value):
                    can_assign = iterable_type.can_assign(right, self)
                    if isinstance(can_assign, CanAssignError):
                        self._show_error_if_checking(
                            source_node,
                            "Unsupported operand for 'in'",
                            ErrorCode.incompatible_argument,
                            detail=str(can_assign),
                        )
                        return TypedValue(bool)
                    else:
                        return TypedValue(bool)

                getitem_result = self._check_dunder_call_or_catch(
                    source_node,
                    left_composite,
                    "__getitem__",
                    [right_composite],
                    allow_call=allow_call,
                )
                if isinstance(getitem_result, Value):
                    return TypedValue(bool)  # Always returns a bool
                self.show_caught_errors(contains_result_or_errors)
                return TypedValue(bool)

            result, _ = self._check_dunder_call(
                source_node,
                left_composite,
                method,
                [right_composite],
                allow_call=allow_call,
            )
            return result

        with self.catch_errors() as left_errors:
            left_result, _ = self._check_dunder_call(
                source_node,
                left_composite,
                method,
                [right_composite],
                allow_call=allow_call,
            )

        with self.catch_errors() as right_errors:
            right_result, _ = self._check_dunder_call(
                source_node,
                right_composite,
                rmethod,
                [left_composite],
                allow_call=allow_call,
            )
        if left_errors:
            if right_errors:
                self.show_error(
                    source_node,
                    f"Unsupported operands for {description}: {left} and {right}",
                    error_code=ErrorCode.unsupported_operation,
                )
                return AnyValue(AnySource.error)
            return right_result
        else:
            if right_errors:
                return left_result
            # The interesting case: neither threw an error. Naively we might
            # want to return the left result, but that fails in a case like
            # this:
            #     df: Any
            #     1 + df
            # because this would return "int" (which is what int.__add__ returns),
            # and "df" might be an object that implements __radd__.
            # Instead, we return Any if that's the right_result. This handles
            # the case above but might return the wrong result in some other rare
            # cases.
            if isinstance(right_result, AnyValue):
                return AnyValue(AnySource.from_another)
            return left_result

    # Indexing

    def visit_Ellipsis(self, node: ast.Ellipsis) -> Value:
        return KnownValue(Ellipsis)

    def visit_Slice(self, node: ast.Slice) -> Value:
        lower = self.visit(node.lower) if node.lower is not None else None
        upper = self.visit(node.upper) if node.upper is not None else None
        step = self.visit(node.step) if node.step is not None else None

        if all(
            val is None or isinstance(val, KnownValue) for val in (lower, upper, step)
        ):
            return KnownValue(
                slice(
                    lower.val if isinstance(lower, KnownValue) else None,
                    upper.val if isinstance(upper, KnownValue) else None,
                    step.val if isinstance(step, KnownValue) else None,
                )
            )
        else:
            return TypedValue(slice)

    # These two are unused in 3.8 and higher, and the typeshed stubs reflect
    # that their .dims and .value attributes don't exist.
    def visit_ExtSlice(self, node: ast.ExtSlice) -> Value:
        # static analysis: ignore[undefined_attribute]
        dims = [self.visit(dim) for dim in node.dims]
        return self._maybe_make_sequence(tuple, dims, node)

    def visit_Index(self, node: ast.Index) -> Value:
        # static analysis: ignore[undefined_attribute]
        return self.visit(node.value)

    # Control flow

    def visit_Await(self, node: ast.Await) -> Value:
        composite = self.composite_from_node(node.value)
        return_value = self.unpack_awaitable(composite, node.value)
        if return_value is NO_RETURN_VALUE:
            self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))
        return return_value

    def unpack_awaitable(self, composite: Composite, node: ast.AST) -> Value:
        tv_map = get_tv_map(AwaitableValue, composite.value, self)
        if isinstance(tv_map, CanAssignError):
            result, _ = self._check_dunder_call(node, composite, "__await__", [])
            return result
        else:
            return tv_map.get(T, AnyValue(AnySource.generic_argument))

    def visit_YieldFrom(self, node: ast.YieldFrom) -> Value:
        self.is_generator = True
        value = self.visit(node.value)
        tv_map = get_tv_map(GeneratorValue, value, self)
        if isinstance(tv_map, CanAssignError):
            can_assign = get_tv_map(AwaitableValue, value, self)
            if not isinstance(can_assign, CanAssignError):
                tv_map = {
                    ReturnT: can_assign.get(T, AnyValue(AnySource.generic_argument))
                }
            else:
                iterable_type = is_iterable(value, self)
                if isinstance(iterable_type, CanAssignError):
                    self._show_error_if_checking(
                        node,
                        f"Cannot use {value} in yield from",
                        error_code=ErrorCode.bad_yield_from,
                        detail=can_assign.display(),
                    )
                    tv_map = {ReturnT: AnyValue(AnySource.error)}
                else:
                    tv_map = {YieldT: iterable_type}

        if self.current_function_info is not None:
            expected_yield = self.current_function_info.get_generator_yield_type(self)
            yield_type = tv_map.get(YieldT, AnyValue(AnySource.generic_argument))
            can_assign = expected_yield.can_assign(yield_type, self)
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    node,
                    f"Cannot yield from {value} (expected {expected_yield})",
                    error_code=ErrorCode.incompatible_yield,
                    detail=can_assign.display(),
                )

            expected_send = self.current_function_info.get_generator_send_type(self)
            send_type = tv_map.get(SendT, AnyValue(AnySource.generic_argument))
            can_assign = send_type.can_assign(expected_send, self)
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    node,
                    (
                        f"Cannot send {send_type} to a generator (expected"
                        f" {expected_send})"
                    ),
                    error_code=ErrorCode.incompatible_yield,
                    detail=can_assign.display(),
                )

        return tv_map.get(ReturnT, AnyValue(AnySource.generic_argument))

    def visit_Yield(self, node: ast.Yield) -> Value:
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
                value = KnownValue(None)

        if node.value is None and self.async_kind in (
            AsyncFunctionKind.normal,
            AsyncFunctionKind.pure,
        ):
            self._show_error_if_checking(node, error_code=ErrorCode.yield_without_value)
        self.is_generator = True

        # unwrap the results of async yields
        if self.async_kind != AsyncFunctionKind.non_async:
            return self._unwrap_yield_result(node, value)
        if self.current_function_info is None:
            return AnyValue(AnySource.inference)
        yield_type = self.current_function_info.get_generator_yield_type(self)
        can_assign = yield_type.can_assign(value, self)
        if isinstance(can_assign, CanAssignError):
            self._show_error_if_checking(
                node,
                (
                    f"Cannot assign value of type {value} to yield expression of type"
                    f" {yield_type}"
                ),
                error_code=ErrorCode.incompatible_yield,
                detail=can_assign.display(),
            )
        return self.current_function_info.get_generator_send_type(self)

    def _unwrap_yield_result(self, node: ast.AST, value: Value) -> Value:
        if isinstance(value, AsyncTaskIncompleteValue):
            return value.value
        elif isinstance(value, TypedValue) and (
            # asynq only supports exactly list and tuple, not subclasses
            # https://github.com/quora/asynq/blob/b07682d8b11e53e4ee5c585020cc9033e239c7eb/asynq/async_task.py#L446
            value.get_type_object().is_exactly({list, tuple})
        ):
            if isinstance(value, SequenceValue) and isinstance(value.typ, type):
                values = [
                    (is_many, self._unwrap_yield_result(node, member))
                    for is_many, member in value.members
                ]
                return SequenceValue.make_or_known(value.typ, values)
            elif isinstance(value, GenericValue):
                member_value = self._unwrap_yield_result(node, value.get_arg(0))
                return GenericValue(value.typ, [member_value])
            else:
                return TypedValue(value.typ)
        elif isinstance(value, TypedValue) and value.get_type_object().is_exactly(
            {dict}
        ):
            if isinstance(value, DictIncompleteValue):
                pairs = [
                    KVPair(
                        pair.key,
                        self._unwrap_yield_result(node, pair.value),
                        pair.is_many,
                        pair.is_required,
                    )
                    for pair in value.kv_pairs
                ]
                return DictIncompleteValue(value.typ, pairs)
            elif isinstance(value, GenericValue):
                val = self._unwrap_yield_result(node, value.get_arg(1))
                return GenericValue(value.typ, [value.get_arg(0), val])
            else:
                return TypedValue(dict)
        elif isinstance(value, KnownValue) and isinstance(value.val, asynq.ConstFuture):
            return KnownValue(value.val.value())
        elif isinstance(value, KnownValue) and value.val is None:
            return value  # we're allowed to yield None
        elif isinstance(value, KnownValue) and isinstance(value.val, (list, tuple)):
            values = [
                self._unwrap_yield_result(node, KnownValue(elt)) for elt in value.val
            ]
            return KnownValue(values)
        elif isinstance(value, AnyValue):
            return AnyValue(AnySource.from_another)
        elif isinstance(value, AnnotatedValue):
            return self._unwrap_yield_result(node, value.value)
        elif isinstance(value, MultiValuedValue):
            return unite_values(
                *[self._unwrap_yield_result(node, val) for val in value.vals]
            )
        elif _is_asynq_future(value):
            return AnyValue(AnySource.inference)
        else:
            self._show_error_if_checking(
                node,
                f"Invalid value yielded: {value}",
                error_code=ErrorCode.bad_async_yield,
            )
            return AnyValue(AnySource.error)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            value = KnownNone
        else:
            value = self.visit(node.value)
        self.return_values.append(value)
        self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))
        if (
            self.expected_return_value is NO_RETURN_VALUE
            and value is not NO_RETURN_VALUE
        ):
            self._show_error_if_checking(
                node, error_code=ErrorCode.no_return_may_return
            )
        elif self.is_generator and self.async_kind == AsyncFunctionKind.non_async:
            if self.current_function_info is not None:
                expected = self.current_function_info.get_generator_return_type(self)
                can_assign = expected.can_assign(value, self)
                if isinstance(can_assign, CanAssignError):
                    self._show_error_if_checking(
                        node,
                        f"Incompatible return value {value} (expected {expected})",
                        error_code=ErrorCode.incompatible_return_value,
                        detail=can_assign.display(),
                    )
        elif self.expected_return_value is not None:
            can_assign = self.expected_return_value.can_assign(value, self)
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    node,
                    (
                        f"Declared return type {self.expected_return_value} is"
                        f" incompatible with actual return type {value}"
                    ),
                    error_code=ErrorCode.incompatible_return_value,
                    detail=can_assign.display(),
                )
        if (
            self.expected_return_value == KnownNone
            and value != KnownNone
            and value is not NO_RETURN_VALUE
        ):
            self._show_error_if_checking(
                node,
                "Function declared as returning None may not return a value",
                error_code=ErrorCode.incompatible_return_value,
            )

    def visit_Raise(self, node: ast.Raise) -> None:
        # we need to record this in the return value so that functions that always raise
        # NotImplementedError aren't inferred as returning None
        self.return_values.append(None)
        self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))

        raised_expr = node.exc

        if raised_expr is not None:
            raised_value = self.visit(raised_expr)
            can_assign = ExceptionValue.can_assign(raised_value, self)
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    node, error_code=ErrorCode.bad_exception, detail=str(can_assign)
                )

        if node.cause is not None:
            cause_value = self.visit(node.cause)
            can_assign = ExceptionOrNone.can_assign(cause_value, self)
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    node,
                    "Invalid object in raise from",
                    error_code=ErrorCode.bad_exception,
                    detail=str(can_assign),
                )

    def visit_Assert(self, node: ast.Assert) -> None:
        test = self._visit_possible_constraint(node.test)
        constraint = extract_constraints(test)
        if node.msg is not None:
            with self.scopes.subscope():
                self.add_constraint(node, constraint.invert())
                self.visit(node.msg)
        self.add_constraint(node, constraint)
        # code after an assert False is unreachable
        boolability = get_boolability(test)
        if boolability is Boolability.value_always_false:
            self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))
        # We don't check value_always_true here; it's fine to have an assertion
        # that the type checker statically thinks is True.
        self._check_boolability(test, node, disabled={ErrorCode.value_always_true})

    def add_constraint(self, node: object, constraint: AbstractConstraint) -> None:
        if constraint is NULL_CONSTRAINT:
            return  # save some work
        self.scopes.current_scope().add_constraint(constraint, node, self.state)

    def _visit_possible_constraint(self, node: ast.AST) -> Value:
        if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            composite = self.composite_from_node(node)
            if composite.varname is not None:
                constraint = Constraint(
                    composite.varname, ConstraintType.is_truthy, True, None
                )
                existing = extract_constraints(composite.value)
                new_value, _ = unannotate_value(composite.value, ConstraintExtension)
                return annotate_with_constraint(
                    new_value, EquivalentConstraint.make([constraint, existing])
                )
            else:
                return composite.value
        else:
            return self.visit(node)

    def visit_Break(self, node: ast.Break) -> None:
        self._set_name_in_scope(LEAVES_LOOP, node, AnyValue(AnySource.marker))

    def visit_Continue(self, node: ast.Continue) -> None:
        self._set_name_in_scope(LEAVES_LOOP, node, AnyValue(AnySource.marker))

    def visit_For(self, node: Union[ast.For, ast.AsyncFor]) -> None:
        iterated_value = self._member_value_of_iterator(
            node.iter, is_async=isinstance(node, ast.AsyncFor)
        )
        if self.options.get_value_for(ForLoopAlwaysEntered):
            always_entered = True
        elif isinstance(iterated_value, Value):
            iterated_value, present = unannotate_value(
                iterated_value, AlwaysPresentExtension
            )
            always_entered = bool(present)
        else:
            always_entered = len(iterated_value) > 0
        if not isinstance(iterated_value, Value):
            iterated_value = unite_and_simplify(
                *iterated_value,
                limit=self.options.get_value_for(UnionSimplificationLimit),
            )
        with self.scopes.subscope() as body_scope:
            with self.scopes.loop_scope():
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
            with self.scopes.subscope():
                with qcore.override(self, "being_assigned", iterated_value):
                    self.visit(node.target)
                self._generic_visit_list(node.body)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        # see comments under For for discussion

        # We don't check boolability here because "while True" is legitimate and common.
        test, constraint = self.constraint_from_condition(
            node.test, check_boolability=False
        )
        always_entered = get_boolability(test) in (
            Boolability.value_always_true,
            Boolability.type_always_true,
        )
        with self.scopes.subscope() as body_scope:
            with self.scopes.loop_scope() as loop_scopes:
                # The "node" argument need not be an AST node but must be unique.
                self.add_constraint((node, 1), constraint)
                self._generic_visit_list(node.body)
        self._handle_loop_else(node.orelse, body_scope, always_entered)

        if self.state == VisitorState.collect_names:
            test, constraint = self.constraint_from_condition(
                node.test, check_boolability=False
            )
            with self.scopes.subscope():
                self.add_constraint((node, 2), constraint)
                self._generic_visit_list(node.body)

        if always_entered and all(LEAVES_LOOP not in scope for scope in loop_scopes):
            # This means the code following the loop is unreachable.
            self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))

    def _handle_loop_else(
        self, orelse: List[ast.stmt], body_scope: SubScope, always_entered: bool
    ) -> None:
        if always_entered:
            self.scopes.combine_subscopes([body_scope])
            # Replace body_scope with a dummy scope, because body_scope
            # should always execute and has already been combined in.
            with self.scopes.subscope() as body_scope:
                pass
        with self.scopes.subscope() as else_scope:
            self._generic_visit_list(orelse)
        self.scopes.combine_subscopes([body_scope, else_scope])

    def _member_value_of_iterator(
        self, node: ast.AST, is_async: bool = False
    ) -> Union[Value, Sequence[Value]]:
        """Analyze an iterator AST node.

        Returns a tuple of two values:
        - A Value object representing a member of the iterator.
        - The number of elements in the iterator, or None if the number is unknown.

        """
        composite = self.composite_from_node(node)
        if is_async:
            iterator, _ = self._check_dunder_call(node, composite, "__aiter__", [])
            anext, _ = self._check_dunder_call(
                node, Composite(iterator, None, node), "__anext__", []
            )
            return self.unpack_awaitable(Composite(anext), node)
        iterated = composite.value
        result = concrete_values_from_iterable(iterated, self)
        if isinstance(result, CanAssignError):
            self._show_error_if_checking(
                node,
                f"{iterated} is not iterable",
                ErrorCode.unsupported_operation,
                detail=str(result),
            )
            return AnyValue(AnySource.error)
        return result

    def visit_With(self, node: ast.With) -> None:
        if len(node.items) == 1:
            with self.scopes.subscope():
                context = self.visit(node.items[0].context_expr)
            if isinstance(context, AnnotatedValue) and context.has_metadata_of_type(
                AssertErrorExtension
            ):
                self._visit_assert_errors_block(node)
                return

        self.visit_single_cm(node.items, node.body, is_async=False)

    def _visit_assert_errors_block(self, node: ast.With) -> None:
        with self.catch_errors() as caught:
            self._generic_visit_list(node.body)
        if not caught:
            self._show_error_if_checking(
                node,
                "No errors found in assert_error() block",
                error_code=ErrorCode.inference_failure,
            )

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_single_cm(node.items, node.body, is_async=True)

    def visit_single_cm(
        self,
        items: List[ast.withitem],
        body: Iterable[ast.AST],
        *,
        is_async: bool = False,
    ) -> None:
        if len(items) == 0:
            self._generic_visit_list(body)
            return
        first_item = items[0]
        can_suppress = self.visit_withitem(first_item, is_async)
        if can_suppress:
            with self.scopes.suppressing_subscope():
                self.visit_single_cm(items[1:], body, is_async=is_async)
        else:
            self.visit_single_cm(items[1:], body, is_async=is_async)

    def visit_withitem(self, node: ast.withitem, is_async: bool = False) -> bool:
        context = self.visit(node.context_expr)
        if is_async:
            protocol = AsyncCustomContextManager
        else:
            protocol = CustomContextManager
        val = GenericValue(protocol, [TypeVarValue(T), TypeVarValue(U)])
        can_assign = get_tv_map(val, context, self)
        if isinstance(can_assign, CanAssignError):
            self._show_error_if_checking(
                node.context_expr,
                f"{context} is not a context manager",
                detail=str(can_assign),
                error_code=ErrorCode.invalid_context_manager,
            )
            assigned = AnyValue(AnySource.error)
            can_suppress = False
        else:
            assigned = can_assign.get(T, AnyValue(AnySource.generic_argument))
            exit_assigned = can_assign.get(U, AnyValue(AnySource.generic_argument))
            exit_boolability = get_boolability(exit_assigned)
            can_suppress = not exit_boolability.is_safely_false()
            if isinstance(exit_assigned, AnyValue) or (
                isinstance(context, TypedValue) and is_context_manager_type(context.typ)
            ):
                # cannot easily infer what the context manager will do,
                # assume it does not suppress exceptions.
                can_suppress = False
        if node.optional_vars is not None:
            with qcore.override(self, "being_assigned", assigned):
                self.visit(node.optional_vars)
        return can_suppress

    def visit_try_except(
        self, node: Union[ast.Try, TryStar], *, is_try_star: bool = False
    ) -> None:
        with self.scopes.subscope():
            with self.scopes.subscope() as dummy_scope:
                pass

            with self.scopes.subscope() as failure_scope:
                with self.scopes.suppressing_subscope() as success_scope:
                    self._generic_visit_list(node.body)

            with self.scopes.subscope() as else_scope:
                self.yield_checker.reset_yield_checks()
                self.scopes.combine_subscopes([success_scope])
                self._generic_visit_list(node.orelse)

            except_scopes = []
            for handler in node.handlers:
                with self.scopes.subscope() as except_scope:
                    except_scopes.append(except_scope)
                    # reset yield checks between branches to avoid incorrect errors when we yield
                    # both in the try and the except block
                    self.yield_checker.reset_yield_checks()
                    # With except*, multiple except* blocks may run, so we need
                    # to combine not just the failure scope, but also the previous
                    # except_scopes.
                    if is_try_star:
                        subscopes = [dummy_scope, failure_scope, *except_scopes]
                    else:
                        subscopes = [dummy_scope, failure_scope]
                    self.scopes.combine_subscopes(subscopes)
                    self.visit(handler)

        self.scopes.combine_subscopes([else_scope, *except_scopes])

    def visit_Try(
        self, node: Union[ast.Try, TryStar], *, is_try_star: bool = False
    ) -> None:
        if node.finalbody:
            with self.scopes.subscope() as failure_scope:
                with self.scopes.suppressing_subscope() as success_scope:
                    self.visit_try_except(node, is_try_star=is_try_star)

            # If the try block fails
            with self.scopes.subscope():
                self.scopes.combine_subscopes([failure_scope])
                self._generic_visit_list(node.finalbody)

            # For the case where execution continues after the try-finally
            self.scopes.combine_subscopes([success_scope])
            self._generic_visit_list(node.finalbody)
        else:
            # Life is much simpler without finally
            self.visit_try_except(node, is_try_star=is_try_star)
        self.yield_checker.reset_yield_checks()

    def visit_TryStar(self, node: TryStar) -> None:
        self.visit_Try(node, is_try_star=True)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            typ = self.visit(node.type)
            is_try_star = not isinstance(self.node_context.contexts[-2], ast.Try)
            possible_types = self._extract_exception_types(
                typ, node, is_try_star=is_try_star
            )
            if node.name is not None:
                to_assign = unite_values(*[typ for _, typ in possible_types])
                if is_try_star:
                    if all(is_exception for is_exception, _ in possible_types):
                        base = ExceptionGroup
                    else:
                        base = BaseExceptionGroup
                    to_assign = GenericValue(base, [to_assign])
                self._set_name_in_scope(node.name, node, value=to_assign, private=True)

        self._generic_visit_list(node.body)

    def _extract_exception_types(
        self, typ: Value, node: ast.AST, is_try_star: bool = False
    ) -> List[Tuple[bool, Value]]:
        possible_types = []
        for subval in flatten_values(typ, unwrap_annotated=True):
            subval = replace_known_sequence_value(subval)
            if isinstance(subval, SequenceValue) and subval.typ is tuple:
                for _, elt in subval.members:
                    possible_types += self._extract_exception_types(
                        elt, node, is_try_star=is_try_star
                    )
                continue
            elif isinstance(subval, GenericValue) and subval.typ is tuple:
                possible_types += self._extract_exception_types(
                    subval.args[0], node, is_try_star=is_try_star
                )
                continue
            elif (
                isinstance(subval, SubclassValue)
                and isinstance(subval.typ, TypedValue)
                and isinstance(subval.typ.typ, type)
            ):
                subval = KnownValue(subval.typ.typ)
            if isinstance(subval, KnownValue):
                if isinstance(subval.val, type) and issubclass(
                    subval.val, BaseException
                ):
                    if is_try_star and issubclass(subval.val, BaseExceptionGroup):
                        self._show_error_if_checking(
                            node,
                            (
                                "ExceptionGroup cannot be used as the type in an"
                                f" except* clause: {subval.val!r}"
                            ),
                            error_code=ErrorCode.bad_except_handler,
                        )
                    is_exception = issubclass(subval.val, Exception)
                    possible_types.append((is_exception, TypedValue(subval.val)))
                else:
                    self._show_error_if_checking(
                        node,
                        f"{subval!r} is not an exception class",
                        error_code=ErrorCode.bad_except_handler,
                    )
                    possible_types.append((False, TypedValue(BaseException)))
            else:
                # TODO consider raising an error for except classes
                # that cannot be statically resolved.
                possible_types.append((False, TypedValue(BaseException)))
        return possible_types

    def visit_If(self, node: ast.If) -> None:
        _, constraint = self.constraint_from_condition(node.test)
        # reset yield checks to avoid incorrect errors when we yield in both the condition and one
        # of the blocks
        self.yield_checker.reset_yield_checks()
        with self.scopes.subscope() as body_scope:
            self.add_constraint(node, constraint)
            self._generic_visit_list(node.body)
        self.yield_checker.reset_yield_checks()

        with self.scopes.subscope() as else_scope:
            self.add_constraint(node, constraint.invert())
            self._generic_visit_list(node.orelse)
        self.scopes.combine_subscopes([body_scope, else_scope])
        self.yield_checker.reset_yield_checks()

    def visit_IfExp(self, node: ast.IfExp) -> Value:
        _, constraint = self.constraint_from_condition(node.test)
        with self.scopes.subscope() as if_scope:
            self.add_constraint(node, constraint)
            then_val = self.visit(node.body)
        with self.scopes.subscope() as else_scope:
            self.add_constraint(node, constraint.invert())
            else_val = self.visit(node.orelse)
        self.scopes.combine_subscopes([if_scope, else_scope])
        return unite_values(then_val, else_val)

    def constraint_from_condition(
        self, node: ast.AST, check_boolability: bool = True
    ) -> Tuple[Value, AbstractConstraint]:
        condition = self._visit_possible_constraint(node)
        constraint = extract_constraints(condition)
        if self._is_collecting():
            return condition, constraint
        if check_boolability:
            disabled = set()
        else:
            disabled = {ErrorCode.type_always_true, ErrorCode.value_always_true}
        self._check_boolability(condition, node, disabled=disabled)
        return condition, constraint

    def _check_boolability(
        self,
        value: Value,
        node: ast.AST,
        *,
        disabled: Container[ErrorCode] = frozenset(),
    ) -> None:
        boolability = get_boolability(value)
        if boolability is Boolability.erroring_bool:
            if ErrorCode.type_does_not_support_bool not in disabled:
                self.show_error(
                    node,
                    f"{value} does not support bool()",
                    error_code=ErrorCode.type_does_not_support_bool,
                )
        elif boolability is Boolability.type_always_true:
            if ErrorCode.type_always_true not in disabled:
                self._show_error_if_checking(
                    node,
                    f"{value} is always True because it does not provide __bool__",
                    error_code=ErrorCode.type_always_true,
                )
        elif boolability in (
            Boolability.value_always_true,
            Boolability.value_always_true_mutable,
        ):
            if ErrorCode.value_always_true not in disabled:
                self.show_error(
                    node,
                    f"{value} is always True",
                    error_code=ErrorCode.value_always_true,
                )

    def visit_Expr(self, node: ast.Expr) -> Value:
        value = self.visit(node.value)
        if _is_asynq_future(value):
            new_node = ast.Expr(value=ast.Yield(value=node.value))
            replacement = self.replace_node(node, new_node)
            self._show_error_if_checking(
                node, error_code=ErrorCode.task_needs_yield, replacement=replacement
            )
        # If the value is an awaitable or is assignable to asyncio.Future, show
        # an error about a missing await.
        elif value.is_type(collections.abc.Awaitable) or value.is_type(asyncio.Future):
            if self.is_async_def:
                new_node = ast.Expr(value=ast.Await(value=node.value))
            else:
                new_node = ast.Expr(value=ast.YieldFrom(value=node.value))
            replacement = self.replace_node(node, new_node)
            self._show_error_if_checking(
                node, error_code=ErrorCode.missing_await, replacement=replacement
            )
        return value

    # Assignments

    def visit_NamedExpr(self, node: NamedExpr) -> Value:
        composite = self.composite_from_walrus(node)
        return composite.value

    def composite_from_walrus(self, node: NamedExpr) -> Composite:
        rhs = self.visit(node.value)
        with qcore.override(self, "being_assigned", rhs):
            if self.in_comprehension_body:
                ctx = self.scopes.ignore_topmost_scope()
            else:
                ctx = qcore.empty_context
            with ctx:
                return self.composite_from_node(node.target)

    def visit_Assign(self, node: ast.Assign) -> None:
        is_yield = isinstance(node.value, ast.Yield)
        value = self.visit(node.value)

        with qcore.override(
            self, "being_assigned", value
        ), self.yield_checker.check_yield_result_assignment(is_yield):
            # syntax like 'x = y = 0' results in multiple targets
            self._generic_visit_list(node.targets)

        if (
            self.current_enum_members is not None
            and self.current_function_name is None
            and isinstance(value, KnownValue)
            and is_hashable(value.val)
        ):
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]

            if value.val in self.current_enum_members:
                self._show_error_if_checking(
                    node,
                    "Duplicate enum member: %s is used for both %s and %s"
                    % (
                        value.val,
                        self.current_enum_members[value.val],
                        ", ".join(names),
                    ),
                    error_code=ErrorCode.duplicate_enum_member,
                )
            else:
                for name in names:
                    self.current_enum_members[value.val] = name

    def is_in_typeddict_definition(self) -> bool:
        return is_instance_of_typing_name(self.current_class, "_TypedDictMeta")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        annotation = self._visit_annotation(node.annotation)
        if isinstance(annotation, KnownValue) and is_typing_name(
            annotation.val, "Final"
        ):
            is_final = True
            expected_type = None
        else:
            expected_type = self._value_of_annotation_type(
                annotation,
                node.annotation,
                is_typeddict=self.is_in_typeddict_definition(),
            )
            # TODO: Also extract Final from more complex annotations
            is_final = False

        if node.value is not None:
            is_yield = isinstance(node.value, ast.Yield)
            value = self.visit(node.value)

            if expected_type is not None:
                can_assign = expected_type.can_assign(value, self)
                if isinstance(can_assign, CanAssignError):
                    self._show_error_if_checking(
                        node,
                        (
                            f"Incompatible assignment: expected {expected_type}, got"
                            f" {value}"
                        ),
                        error_code=ErrorCode.incompatible_assignment,
                        detail=can_assign.display(),
                    )
                # We set the declared type on initial assignment, so that the
                # annotation can be used to adjust pyanalyze's type inference.
                value = expected_type

        else:
            is_yield = False
            value = None

        with qcore.override(
            self, "being_assigned", value
        ), self.yield_checker.check_yield_result_assignment(is_yield), qcore.override(
            self, "ann_assign_type", (expected_type, is_final)
        ):
            self.visit(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        is_yield = isinstance(node.value, ast.Yield)
        rhs = self.composite_from_node(node.value)

        if isinstance(node.target, ast.Name):
            lhs = self.composite_from_name(node.target, force_read=True)
        else:
            lhs = Composite(AnyValue(AnySource.inference), None, node.target)

        value = self._visit_binop_internal(
            node.target, lhs, node.op, node.value, rhs, node, is_inplace=True
        )

        with qcore.override(
            self, "being_assigned", value
        ), self.yield_checker.check_yield_result_assignment(is_yield):
            # syntax like 'x = y = 0' results in multiple targets
            self.visit(node.target)

    def visit_Name(self, node: ast.Name, force_read: bool = False) -> Value:
        return self.composite_from_name(node, force_read=force_read).value

    def composite_from_name(
        self, node: ast.Name, force_read: bool = False
    ) -> Composite:
        if force_read or self._is_read_ctx(node.ctx):
            self.yield_checker.record_usage(node.id, node)
            value, origin = self.resolve_name(node)
            varname_value = self.checker.maybe_get_variable_name_value(node.id)
            if varname_value is not None and self._should_use_varname_value(value):
                value = varname_value
            self.check_deprecation(node, value)
            return Composite(value, VarnameWithOrigin(node.id, origin), node)
        elif self._is_write_ctx(node.ctx):
            if self._name_node_to_statement is not None:
                statement = self.node_context.nearest_enclosing(
                    (ast.stmt, ast.comprehension)
                )
                self._name_node_to_statement[node] = statement

            value = self.being_assigned
            if (
                value is None
                and self.scopes.current_scope().scope_type == ScopeType.class_scope
                and self.ann_assign_type is not None
            ):
                ann_assign_type, _ = self.ann_assign_type
                if ann_assign_type is not None:
                    value = ann_assign_type
            if value is not None:
                self.yield_checker.record_assignment(node.id)
                value, origin = self._set_name_in_scope(node.id, node, value=value)
            else:
                value = AnyValue(AnySource.inference)
                origin = EMPTY_ORIGIN
            varname = VarnameWithOrigin(node.id, origin)
            constraint = Constraint(varname, ConstraintType.is_truthy, True, None)
            value = annotate_with_constraint(value, constraint)
            return Composite(value, varname, node)
        else:
            # not sure when (if ever) the other contexts can happen
            self.show_error(node, f"Bad context: {node.ctx}", ErrorCode.unexpected_node)
            return Composite(AnyValue(AnySource.error), None, node)

    def visit_Starred(self, node: ast.Starred) -> Value:
        val = self.visit(node.value)
        return _StarredValue(val, node.value)

    def visit_arg(self, node: ast.arg) -> None:
        self.yield_checker.record_assignment(node.arg)
        # it's none only for AnnAssign nodes without a value
        assert self.being_assigned is not None, "should not happen"
        self._set_name_in_scope(node.arg, node, value=self.being_assigned)

    def _should_use_varname_value(self, value: Value) -> bool:
        """Returns whether a value should be replaced with VariableNameValue.

        VariableNameValues are used for things like uids that are represented as integers, but
        in places where we don't necessarily have precise annotations. Therefore, we replace
        only AnyValue.

        """
        return (
            isinstance(value, AnyValue) and value.source is not AnySource.variable_name
        )

    def visit_Subscript(self, node: ast.Subscript) -> Value:
        return self.composite_from_subscript(node).value

    def composite_from_subscript(self, node: ast.Subscript) -> Composite:
        root_composite = self.composite_from_node(node.value)
        index_composite = self.composite_from_node(node.slice)
        index = index_composite.value
        if (
            root_composite.varname is not None
            and isinstance(index, KnownValue)
            and is_hashable(index.val)
        ):
            varname = self._extend_composite(root_composite, index, node)
        else:
            varname = None
        if isinstance(root_composite.value, MultiValuedValue):
            values = [
                self._composite_from_subscript_no_mvv(
                    node,
                    Composite(val, root_composite.varname, root_composite.node),
                    index_composite,
                    varname,
                )
                for val in root_composite.value.vals
            ]
            return_value = unite_values(*values)
        else:
            return_value = self._composite_from_subscript_no_mvv(
                node, root_composite, index_composite, varname
            )
        return Composite(return_value, varname, node)

    def _composite_from_subscript_no_mvv(
        self,
        node: ast.Subscript,
        root_composite: Composite,
        index_composite: Composite,
        composite_var: Optional[VarnameWithOrigin],
    ) -> Value:
        value = root_composite.value
        index = index_composite.value

        if isinstance(node.ctx, ast.Store):
            if self.ann_assign_type is not None:
                self._show_error_if_checking(
                    node, error_code=ErrorCode.invalid_annotated_assignment
                )
            if self.being_assigned is None:
                assert (
                    self.ann_assign_type is not None
                ), "should only happen in AnnAssign"
                return AnyValue(AnySource.inference)
            if (
                composite_var is not None
                and self.scopes.scope_type() == ScopeType.function_scope
            ):
                self.scopes.set(
                    composite_var.get_varname(), self.being_assigned, node, self.state
                )
            self._check_dunder_call(
                node.value,
                root_composite,
                "__setitem__",
                [index_composite, Composite(self.being_assigned, None, node)],
            )
            return self.being_assigned
        elif isinstance(node.ctx, ast.Load):
            if sys.version_info >= (3, 9) and value == KnownValue(type):
                # In Python 3.9+ "type[int]" is legal, but neither
                # type.__getitem__ nor type.__class_getitem__ exists at runtime. Support
                # it directly instead.
                if isinstance(index, KnownValue):
                    # self-check throws an error in 3.8 and lower
                    # static analysis: ignore[unsupported_operation]
                    return_value = KnownValue(type[index.val])
                else:
                    return_value = AnyValue(AnySource.inference)
            else:
                with self.catch_errors():
                    getitem = self._get_dunder(node.value, value, "__getitem__")
                if getitem is not UNINITIALIZED_VALUE:
                    return_value = self.check_call(
                        node.value,
                        getitem,
                        [root_composite, index_composite],
                        allow_call=True,
                    )
                else:
                    # If there was no __getitem__, try __class_getitem__ in 3.7+
                    cgi = self.get_attribute(
                        Composite(value), "__class_getitem__", node.value
                    )
                    if cgi is UNINITIALIZED_VALUE:
                        self._show_error_if_checking(
                            node,
                            f"Object {value} does not support subscripting",
                            error_code=ErrorCode.unsupported_operation,
                        )
                        return_value = AnyValue(AnySource.error)
                    else:
                        return_value = self.check_call(
                            node.value, cgi, [index_composite], allow_call=True
                        )

                if (
                    self._should_use_varname_value(return_value)
                    and isinstance(index, KnownValue)
                    and isinstance(index.val, str)
                ):
                    varname_value = self.checker.maybe_get_variable_name_value(
                        index.val
                    )
                    if varname_value is not None:
                        return_value = varname_value

            if (
                composite_var is not None
                and self.scopes.scope_type() == ScopeType.function_scope
            ):
                local_value = self._get_composite(
                    composite_var.get_varname(), node, return_value
                )
                if local_value is not UNINITIALIZED_VALUE:
                    return_value = local_value
            return return_value
        elif isinstance(node.ctx, ast.Del):
            result, _ = self._check_dunder_call(
                node.value, root_composite, "__delitem__", [index_composite]
            )
            return result
        else:
            self.show_error(
                node,
                f"Unexpected subscript context: {node.ctx}",
                ErrorCode.unexpected_node,
            )
            return AnyValue(AnySource.error)

    def _get_dunder(self, node: ast.AST, callee_val: Value, method_name: str) -> Value:
        lookup_val = callee_val.get_type_value()
        method_object = self.get_attribute(
            Composite(lookup_val),
            method_name,
            node,
            ignore_none=self.options.get_value_for(IgnoreNoneAttributes),
        )
        if method_object is UNINITIALIZED_VALUE:
            self.show_error(
                node,
                f"Object of type {callee_val} does not support {method_name!r}",
                error_code=ErrorCode.unsupported_operation,
            )
        return method_object

    def _check_dunder_call_or_catch(
        self,
        node: ast.AST,
        callee_composite: Composite,
        method_name: str,
        args: Iterable[Composite],
        allow_call: bool = False,
    ) -> Union[Value, List[node_visitor.Error]]:
        """Use this for checking a dunder call that may fall back to another.

        There are three cases:
        - The dunder does not exist. We want to defer the error, in case the fallback
          exists.
        - The dunder exists and the call typechecks. We want to return its result.
        - The dunder exists, but the call doesn't typecheck. We want to show the error
          immediately and return Any.

        """
        with self.catch_errors() as errors:
            result, exists = self._check_dunder_call(
                node, callee_composite, method_name, args, allow_call=allow_call
            )
        if not errors:
            return result
        elif exists:
            # Inplace method exists, but it doesn't accept these arguments
            self.show_caught_errors(errors)
            return result
        else:
            return errors

    def _check_dunder_call(
        self,
        node: ast.AST,
        callee_composite: Composite,
        method_name: str,
        args: Iterable[Composite],
        allow_call: bool = False,
    ) -> Tuple[Value, bool]:
        if isinstance(callee_composite.value, MultiValuedValue):
            composites = [
                Composite(val, callee_composite.varname, callee_composite.node)
                for val in callee_composite.value.vals
            ]
            with qcore.override(self, "in_union_decomposition", True):
                values_and_exists = [
                    self._check_dunder_call_no_mvv(
                        node, composite, method_name, args, allow_call
                    )
                    for composite in composites
                ]
            values = [value for value, _ in values_and_exists]
            # TODO: We should do something more complex when unions are involved.
            exists = all(exists for _, exists in values_and_exists)
            return (
                unite_and_simplify(
                    *values, limit=self.options.get_value_for(UnionSimplificationLimit)
                ),
                exists,
            )
        return self._check_dunder_call_no_mvv(
            node, callee_composite, method_name, args, allow_call
        )

    def _check_dunder_call_no_mvv(
        self,
        node: ast.AST,
        callee_composite: Composite,
        method_name: str,
        args: Iterable[Composite],
        allow_call: bool = False,
    ) -> Tuple[Value, bool]:
        method_object = self._get_dunder(node, callee_composite.value, method_name)
        if method_object is UNINITIALIZED_VALUE:
            return AnyValue(AnySource.error), False
        return_value = self.check_call(
            node, method_object, [callee_composite, *args], allow_call=allow_call
        )
        return return_value, True

    def _get_composite(self, composite: Varname, node: ast.AST, value: Value) -> Value:
        local_value, _ = self.scopes.current_scope().get_local(
            composite, node, self.state, fallback_value=value
        )
        if isinstance(local_value, MultiValuedValue):
            vals = [val for val in local_value.vals if val is not UNINITIALIZED_VALUE]
            if vals:
                return unite_values(*vals)
            else:
                return NO_RETURN_VALUE
        return local_value

    def visit_Attribute(self, node: ast.Attribute) -> Value:
        return self.composite_from_attribute(node).value

    def _extend_composite(
        self, root_composite: Composite, index: CompositeIndex, node: ast.AST
    ) -> Optional[VarnameWithOrigin]:
        varname = root_composite.get_extended_varname(index)
        if varname is None:
            return None
        origin = self.scopes.current_scope().get_origin(varname, node, self.state)
        return root_composite.get_extended_varname_with_origin(index, origin)

    def composite_from_attribute(self, node: ast.Attribute) -> Composite:
        if isinstance(node.value, ast.Name):
            attr_str = f"{node.value.id}.{node.attr}"
            if self._is_write_ctx(node.ctx):
                self.yield_checker.record_assignment(attr_str)
            else:
                self.yield_checker.record_usage(attr_str, node)

        root_composite = self.composite_from_node(node.value)
        composite = self._extend_composite(root_composite, node.attr, node)
        if self._is_write_ctx(node.ctx):
            # TODO: We should do something here if we're in an AnnAssign, e.g.
            # note the type in the class's namespace.
            if self.being_assigned is None:
                assert (
                    self.ann_assign_type is not None
                ), "should only happen in AnnAssign"
                return Composite(AnyValue(AnySource.inference), composite, node)
            if (
                composite is not None
                and self.scopes.scope_type() == ScopeType.function_scope
            ):
                self.scopes.set(
                    composite.get_varname(), self.being_assigned, node, self.state
                )

            if isinstance(root_composite.value, TypedValue):
                typ = root_composite.value.typ
                if isinstance(typ, type):
                    self._record_type_attr_set(
                        typ, node.attr, node, self.being_assigned
                    )
            return Composite(self.being_assigned, composite, node)
        elif self._is_read_ctx(node.ctx):
            if self._is_checking():
                if (
                    isinstance(root_composite.value, KnownValue)
                    and isinstance(root_composite.value.val, types.ModuleType)
                    and root_composite.value.val.__name__ is not None
                ):
                    self.reexport_tracker.record_attribute_accessed(
                        root_composite.value.val.__name__, node.attr, node, self
                    )
            value = self.get_attribute(
                root_composite,
                node.attr,
                node,
                use_fallback=True,
                ignore_none=self.options.get_value_for(IgnoreNoneAttributes),
            )
            self.check_deprecation(node, value)
            if self._should_use_varname_value(value):
                varname_value = self.checker.maybe_get_variable_name_value(node.attr)
                if varname_value is not None:
                    return Composite(varname_value, composite, node)
            if (
                composite is not None
                and self.scopes.scope_type() == ScopeType.function_scope
            ):
                local_value = self._get_composite(composite.get_varname(), node, value)
                if local_value is not UNINITIALIZED_VALUE:
                    value = local_value
            return Composite(value, composite, node)
        else:
            self.show_error(node, "Unknown context", ErrorCode.unexpected_node)
            return Composite(AnyValue(AnySource.error), composite, node)

    def get_attribute(
        self,
        root_composite: Composite,
        attr: str,
        node: Optional[ast.AST] = None,
        *,
        ignore_none: bool = False,
        use_fallback: bool = False,
        prefer_typeshed: bool = False,
    ) -> Value:
        """Get an attribute of this value.

        Returns :data:`pyanalyze.value.UNINITIALIZED_VALUE` if the attribute cannot be found.

        """
        if isinstance(root_composite.value, TypeVarValue):
            root_composite = Composite(
                value=root_composite.value.get_fallback_value(),
                varname=root_composite.varname,
                node=root_composite.node,
            )
        if is_union(root_composite.value):
            results = []
            for subval in flatten_values(root_composite.value):
                composite = Composite(
                    subval, root_composite.varname, root_composite.node
                )
                subresult = self.get_attribute(
                    composite,
                    attr,
                    node,
                    ignore_none=ignore_none,
                    use_fallback=use_fallback,
                )
                if (
                    subresult is UNINITIALIZED_VALUE
                    and use_fallback
                    and node is not None
                ):
                    subresult = self._get_attribute_fallback(subval, attr, node)
                results.append(subresult)
            return unite_values(*results)
        ctx = _AttrContext(
            root_composite,
            attr,
            self,
            node=node,
            ignore_none=ignore_none,
            prefer_typeshed=prefer_typeshed,
        )
        result = attributes.get_attribute(ctx)
        if result is UNINITIALIZED_VALUE and use_fallback and node is not None:
            return self._get_attribute_fallback(root_composite.value, attr, node)
        return result

    def get_attribute_from_value(
        self, root_value: Value, attribute: str, *, prefer_typeshed: bool = False
    ) -> Value:
        return self.get_attribute(
            Composite(root_value), attribute, prefer_typeshed=prefer_typeshed
        )

    def _get_attribute_fallback(
        self, root_value: Value, attr: str, node: ast.AST
    ) -> Value:
        # We don't throw an error in many
        # cases where we're not quite sure whether an attribute
        # will exist.
        if isinstance(root_value, AnnotatedValue):
            root_value = root_value.value
        if isinstance(root_value, UnboundMethodValue):
            if self._should_ignore_val(node):
                return AnyValue(AnySource.error)
        elif isinstance(root_value, KnownValue):
            # super calls on mixin classes may use attributes that are defined only on child classes
            if isinstance(root_value.val, super):
                subclasses = qcore.inspection.get_subclass_tree(
                    root_value.val.__thisclass__
                )
                if any(
                    hasattr(cls, attr)
                    for cls in subclasses
                    if cls is not root_value.val.__thisclass__
                ):
                    return AnyValue(AnySource.inference)

            # Ignore objects that override __getattr__
            if not self._has_only_known_attributes(root_value.val) and (
                _static_hasattr(root_value.val, "__getattr__")
                or self._should_ignore_val(node)
            ):
                return AnyValue(AnySource.inference)
        elif isinstance(root_value, TypedValue):
            root_type = root_value.typ
            if isinstance(root_type, type) and not self._has_only_known_attributes(
                root_type
            ):
                return self._maybe_get_attr_value(root_type, attr)
        elif isinstance(root_value, SubclassValue):
            if isinstance(root_value.typ, TypedValue):
                root_type = root_value.typ.typ
                if isinstance(root_type, type) and not self._has_only_known_attributes(
                    root_type
                ):
                    return self._maybe_get_attr_value(root_type, attr)
            else:
                return AnyValue(AnySource.inference)
        elif isinstance(root_value, MultiValuedValue):
            return unite_values(
                *[
                    self._get_attribute_fallback(val, attr, node)
                    for val in root_value.vals
                ]
            )
        self._show_error_if_checking(
            node,
            f"{root_value} has no attribute {attr!r}",
            ErrorCode.undefined_attribute,
        )
        return AnyValue(AnySource.error)

    def _has_only_known_attributes(self, typ: object) -> bool:
        if not isinstance(typ, type):
            return False
        if issubclass(typ, tuple) and not hasattr(typ, "__getattr__"):
            # namedtuple
            return True
        if issubclass(typ, enum.Enum):
            return True
        ts_finder = self.checker.ts_finder
        if (
            (ts_finder.has_stubs(typ) or is_dataclass_type(typ))
            and not ts_finder.has_attribute(typ, "__getattr__")
            and not ts_finder.has_attribute(typ, "__getattribute__")
            and not attributes.may_have_dynamic_attributes(typ)
            and not hasattr(typ, "__getattr__")
        ):
            return True
        return False

    def composite_from_node(self, node: ast.AST) -> Composite:
        if isinstance(node, ast.Attribute):
            composite = self.composite_from_attribute(node)
        elif isinstance(node, ast.Name):
            composite = self.composite_from_name(node)
        elif isinstance(node, ast.Subscript):
            composite = self.composite_from_subscript(node)
        elif isinstance(node, ast.Index):
            # static analysis: ignore[undefined_attribute]
            composite = self.composite_from_node(node.value)
        elif isinstance(node, (ast.ExtSlice, ast.Slice)):
            # These don't have a .lineno attribute, which would otherwise cause trouble.
            composite = Composite(self.visit(node), None, None)
        # We need better support for version-straddling code
        # static analysis: ignore[value_always_true]
        elif hasattr(ast, "NamedExpr") and isinstance(node, ast.NamedExpr):
            composite = self.composite_from_walrus(node)
        else:
            composite = Composite(self.visit(node), None, node)
        if self.annotate:
            node.inferred_value = composite.value
        return composite

    def varname_for_constraint(self, node: ast.AST) -> Optional[VarnameWithOrigin]:
        """Given a node, returns a variable name that could be used in a local scope."""
        composite = self.composite_from_node(node)
        return composite.varname

    def varname_for_self_constraint(self, node: ast.AST) -> Optional[VarnameWithOrigin]:
        """Helper for constraints on self from method calls.

        Given an ``ast.Call`` node representing a method call, return the variable name
        to be used for a constraint on the self object.

        """
        if not isinstance(node, ast.Call):
            return None
        if isinstance(node.func, ast.Attribute):
            return self.varname_for_constraint(node.func.value)
        else:
            return None

    def _should_ignore_val(self, node: ast.AST) -> bool:
        if node is not None:
            path = get_attribute_path(node)
            if path is not None:
                ignored_paths = self.options.get_value_for(IgnoredPaths)
                for ignored_path in ignored_paths:
                    if path[: len(ignored_path)] == ignored_path:
                        return True
                if path[-1] in self.options.get_value_for(IgnoredEndOfReference):
                    self.log(logging.INFO, "Ignoring end of reference", path)
                    return True
        return False

    # Call nodes

    def visit_keyword(self, node: ast.keyword) -> Tuple[Optional[str], Composite]:
        return (node.arg, self.composite_from_node(node.value))

    def visit_Call(self, node: ast.Call) -> Value:
        callee_wrapped = self.visit(node.func)
        args = [self.composite_from_node(arg) for arg in node.args]
        if node.keywords:
            keywords = [self.visit_keyword(kw) for kw in node.keywords]
        else:
            keywords = []

        return_value = self.check_call(
            node, callee_wrapped, args, keywords, allow_call=self.in_annotation
        )

        if self._is_checking():
            self.yield_checker.record_call(callee_wrapped, node)
            self.asynq_checker.check_call(callee_wrapped, node)

        if self.collector is not None:
            callee_val = None
            if isinstance(callee_wrapped, UnboundMethodValue):
                callee_val = callee_wrapped.get_method()
            elif isinstance(callee_wrapped, KnownValue):
                callee_val = callee_wrapped.val
            elif isinstance(callee_wrapped, SubclassValue) and isinstance(
                callee_wrapped.typ, TypedValue
            ):
                callee_val = callee_wrapped.typ.typ

            if callee_val is not None:
                caller = (
                    self.current_function
                    if self.current_function is not None
                    else self.module
                )
                if caller is not None:
                    self.collector.record_call(caller, callee_val)

        if (
            isinstance(callee_wrapped, KnownValue)
            and callee_wrapped.val is assert_error
        ):
            return annotate_value(return_value, [AssertErrorExtension()])
        return return_value

    def _can_perform_call(
        self, args: Iterable[Value], keywords: Iterable[Tuple[Optional[str], Value]]
    ) -> Annotated[
        bool,
        ParameterTypeGuard["args", Iterable[KnownValue]],
        ParameterTypeGuard["keywords", Iterable[Tuple[str, KnownValue]]],
    ]:
        """Returns whether all of the arguments were inferred successfully."""
        return all(isinstance(arg, KnownValue) for arg in args) and all(
            keyword is not None and isinstance(arg, KnownValue)
            for keyword, arg in keywords
        )

    def check_call(
        self,
        node: Optional[ast.AST],
        callee: Value,
        args: Iterable[Composite],
        keywords: Iterable[Tuple[Optional[str], Composite]] = (),
        *,
        allow_call: bool = False,
    ) -> Value:
        if isinstance(callee, MultiValuedValue):
            with qcore.override(self, "in_union_decomposition", True):
                values = [
                    self._check_call_no_mvv(
                        node, val, args, keywords, allow_call=allow_call
                    )
                    for val in callee.vals
                ]

            pairs = [
                unannotate_value(val, NoReturnConstraintExtension) for val in values
            ]
            val = unite_values(*[val for val, _ in pairs])
            constraint = OrConstraint.make(
                [
                    AndConstraint.make(ext.constraint for ext in exts)
                    for _, exts in pairs
                ]
            )
        else:
            val = self._check_call_no_mvv(
                node, callee, args, keywords, allow_call=allow_call
            )
            val, nru_extensions = unannotate_value(val, NoReturnConstraintExtension)
            constraint = AndConstraint.make(ext.constraint for ext in nru_extensions)
        self.add_constraint(node, constraint)
        return val

    def _check_call_no_mvv(
        self,
        node: Optional[ast.AST],
        callee_wrapped: Value,
        args: Iterable[Composite],
        keywords: Iterable[Tuple[Optional[str], Composite]] = (),
        *,
        allow_call: bool = False,
    ) -> Value:
        if isinstance(callee_wrapped, KnownValue) and any(
            callee_wrapped.val is ignored
            for ignored in self.options.get_value_for(IgnoredCallees)
        ):
            self.log(logging.INFO, "Ignoring callee", callee_wrapped)
            return AnyValue(AnySource.error)

        extended_argspec = self.signature_from_value(callee_wrapped, node)
        if extended_argspec is ANY_SIGNATURE:
            # don't bother calling it
            extended_argspec = None
            return_value = AnyValue(AnySource.from_another)

        elif extended_argspec is None:
            if node is not None:
                self._show_error_if_checking(
                    node,
                    f"{callee_wrapped} is not callable",
                    error_code=ErrorCode.not_callable,
                )
            return_value = AnyValue(AnySource.error)

        else:
            arguments = [
                (Composite(arg.value.value, arg.varname, arg.node), ARGS)
                if isinstance(arg.value, _StarredValue)
                else (arg, None)
                for arg in args
            ] + [
                (value, KWARGS) if keyword is None else (value, keyword)
                for keyword, value in keywords
            ]
            if self._is_checking():
                return_value = extended_argspec.check_call(arguments, self, node)
            else:
                with self.catch_errors():
                    return_value = extended_argspec.check_call(arguments, self, node)

        if extended_argspec is not None and not extended_argspec.has_return_value():
            local = self.get_local_return_value(extended_argspec)
            if local is not None:
                return_value = local

        if allow_call and isinstance(callee_wrapped, KnownValue):
            arg_values = [arg.value for arg in args]
            kw_values = [(kw, composite.value) for kw, composite in keywords]
            if self._can_perform_call(arg_values, kw_values):
                try:
                    result = callee_wrapped.val(
                        *[arg.val for arg in arg_values],
                        **{key: value.val for key, value in kw_values},
                    )
                except Exception as e:
                    self.log(logging.INFO, "exception calling", (callee_wrapped, e))
                else:
                    if result is NotImplemented:
                        self.show_error(
                            node,
                            f"Call to {callee_wrapped.val} is not supported",
                            error_code=ErrorCode.incompatible_call,
                        )
                    return_value = KnownValue(result)

        if return_value is NO_RETURN_VALUE and node is not None:
            self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))

        # for .asynq functions, we use the argspec for the underlying function, but that means
        # that the return value is not wrapped in AsyncTask, so we do that manually here
        if isinstance(callee_wrapped, KnownValue) and is_dot_asynq_function(
            callee_wrapped.val
        ):
            async_fn = callee_wrapped.val.__self__
            return AsyncTaskIncompleteValue(_get_task_cls(async_fn), return_value)
        elif isinstance(
            callee_wrapped, UnboundMethodValue
        ) and callee_wrapped.secondary_attr_name in ("async", "asynq"):
            async_fn = callee_wrapped.get_method()
            return AsyncTaskIncompleteValue(_get_task_cls(async_fn), return_value)
        elif isinstance(callee_wrapped, UnboundMethodValue) and asynq.is_pure_async_fn(
            callee_wrapped.get_method()
        ):
            return return_value
        else:
            if (
                isinstance(return_value, AnyValue)
                and isinstance(callee_wrapped, KnownValue)
                and asynq.is_pure_async_fn(callee_wrapped.val)
            ):
                task_cls = _get_task_cls(callee_wrapped.val)
                if isinstance(task_cls, type):
                    return TypedValue(task_cls)
            return return_value

    def signature_from_value(
        self, value: Value, node: Optional[ast.AST] = None
    ) -> MaybeSignature:
        def get_call_attribute(value: Value) -> Value:
            return self.get_attribute(
                Composite(value),
                "__call__",
                node,
                ignore_none=self.options.get_value_for(IgnoreNoneAttributes),
            )

        return self.checker.signature_from_value(
            value,
            get_return_override=self.get_local_return_value,
            get_call_attribute=get_call_attribute,
        )

    # Match statements

    def visit_Match(self, node: Match) -> None:
        subject = self.composite_from_node(node.subject)
        patma_visitor = PatmaVisitor(self)
        with qcore.override(self, "match_subject", subject):
            constraints_to_apply = []
            subscopes = []
            for case in node.cases:
                with self.scopes.subscope() as case_scope:
                    for constraint in constraints_to_apply:
                        self.add_constraint(case, constraint)
                    self.match_subject = self.match_subject._replace(
                        value=constrain_value(
                            self.match_subject.value,
                            AndConstraint.make(constraints_to_apply),
                        )
                    )

                    pattern_constraint = patma_visitor.visit(case.pattern)
                    constraints = [pattern_constraint]
                    self.add_constraint(case.pattern, pattern_constraint)
                    if case.guard is not None:
                        _, guard_constraint = self.constraint_from_condition(case.guard)
                        self.add_constraint(case.guard, guard_constraint)
                        constraints.append(guard_constraint)

                    constraints_to_apply.append(
                        AndConstraint.make(constraints).invert()
                    )
                    self._generic_visit_list(case.body)
                    subscopes.append(case_scope)

                self.yield_checker.reset_yield_checks()

            with self.scopes.subscope() as else_scope:
                for constraint in constraints_to_apply:
                    self.add_constraint(node, constraint)
                subscopes.append(else_scope)
            self.scopes.combine_subscopes(subscopes)

    # Attribute checking

    def _record_class_examined(self, cls: type) -> None:
        if self.attribute_checker is not None:
            self.attribute_checker.record_class_examined(cls)

    def _record_type_has_dynamic_attrs(self, typ: type) -> None:
        if self.attribute_checker is not None:
            self.attribute_checker.record_type_has_dynamic_attrs(typ)

    def _record_type_attr_set(
        self, typ: type, attr_name: str, node: ast.AST, value: Value
    ) -> None:
        if self.attribute_checker is not None:
            self.attribute_checker.record_attribute_set(typ, attr_name, node, value)

    def _record_type_attr_read(self, typ: type, attr_name: str, node: ast.AST) -> None:
        if self.attribute_checker is not None:
            self.attribute_checker.record_attribute_read(typ, attr_name, node, self)

    def _maybe_get_attr_value(self, typ: type, attr_name: str) -> Value:
        if self.attribute_checker is not None:
            return self.attribute_checker.get_attribute_value(typ, attr_name)
        else:
            return AnyValue(AnySource.inference)

    # Finding unused objects

    def _maybe_record_usage(
        self, module_or_class: object, attribute: str, value: Value
    ) -> None:
        if self.unused_finder is None:
            return

        # in this case class isn't available
        if self.scopes.scope_type() == ScopeType.function_scope and self._is_checking():
            return

        if isinstance(value, KnownValue) and self.current_class is not None:
            # exclude calls within a class (probably in super calls)
            if value.val is self.current_class:
                return

            inner = UnwrapClass.unwrap(value.val, self.options)
            if inner is self.current_class:
                return

        if self.module is not None and isinstance(module_or_class, types.ModuleType):
            self.unused_finder.record(module_or_class, attribute, self.module.__name__)

    @classmethod
    def _get_argument_parser(cls) -> ArgumentParser:
        parser = super()._get_argument_parser()
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
        parser.add_argument(
            "--config-file",
            type=Path,
            help="Path to a pyproject.toml configuration file",
        )
        parser.add_argument(
            "--display-options",
            action="store_true",
            default=False,
            help="Display the options used for this check, then exit",
        )
        add_arguments(parser)
        return parser

    @classmethod
    def get_description_for_error_code(cls, error_code: ErrorCode) -> str:
        return ERROR_DESCRIPTION[error_code]

    @classmethod
    def get_default_directories(
        cls, checker: Checker, **kwargs: Any
    ) -> Tuple[str, ...]:
        paths = checker.options.get_value_for(Paths)
        return tuple(str(path) for path in paths)

    @classmethod
    def _get_default_settings(cls) -> Optional[Dict[enum.Enum, bool]]:
        return {}

    @classmethod
    def prepare_constructor_kwargs(
        cls, kwargs: Mapping[str, Any], extra_options: Sequence[ConfigOption] = ()
    ) -> Mapping[str, Any]:
        kwargs = dict(kwargs)
        instances = [*extra_options]
        if "settings" in kwargs:
            for error_code, value in kwargs["settings"].items():
                option_cls = ConfigOption.registry[error_code.name]
                instances.append(option_cls(value, from_command_line=True))
        files = kwargs.pop("files", [])
        if files:
            instances.append(Paths(files, from_command_line=True))
        for name, option_cls in ConfigOption.registry.items():
            if not option_cls.should_create_command_line_option:
                continue
            if name not in kwargs:
                continue
            value = kwargs.pop(name)
            instances.append(option_cls(value, from_command_line=True))
        config_file = kwargs.pop("config_file", None)
        if config_file is None:
            config_filename = cls.config_filename
            if config_filename is not None:
                module_path = Path(sys.modules[cls.__module__].__file__).parent
                config_file = module_path / config_filename
        options = Options.from_option_list(instances, config_file_path=config_file)
        if kwargs.pop("display_options", False):
            options.display()
            sys.exit(0)
        kwargs.setdefault("checker", Checker(raw_options=options))
        patch_typing_overload()
        return kwargs

    def is_enabled(self, error_code: enum.Enum) -> bool:
        if not isinstance(error_code, ErrorCode):
            return False
        return self.options.is_error_code_enabled(error_code)

    @classmethod
    def perform_final_checks(
        cls, kwargs: Mapping[str, Any]
    ) -> List[node_visitor.Failure]:
        return kwargs["checker"].perform_final_checks()

    @classmethod
    def _run_on_files(
        cls,
        files: List[str],
        *,
        checker: Checker,
        find_unused: bool = False,
        find_unused_attributes: bool = False,
        attribute_checker: Optional[ClassAttributeChecker] = None,
        unused_finder: Optional[UnusedObjectFinder] = None,
        **kwargs: Any,
    ) -> List[node_visitor.Failure]:
        attribute_checker_enabled = checker.options.is_error_code_enabled_anywhere(
            ErrorCode.attribute_is_never_set
        )
        if attribute_checker is None:
            inner_attribute_checker_obj = attribute_checker = ClassAttributeChecker(
                enabled=attribute_checker_enabled,
                should_check_unused_attributes=find_unused_attributes,
                should_serialize=kwargs.get("parallel", False),
                options=checker.options,
            )
        else:
            inner_attribute_checker_obj = qcore.empty_context
        if unused_finder is None:
            unused_finder = UnusedObjectFinder(
                checker.options,
                enabled=find_unused or checker.options.get_value_for(EnforceNoUnused),
                print_output=False,
            )
        with inner_attribute_checker_obj as inner_attribute_checker:
            with unused_finder as inner_unused_finder:
                all_failures = super()._run_on_files(
                    files,
                    attribute_checker=attribute_checker
                    if attribute_checker is not None
                    else inner_attribute_checker,
                    unused_finder=inner_unused_finder,
                    checker=checker,
                    **kwargs,
                )
        if unused_finder is not None:
            for unused_object in unused_finder.get_unused_objects():
                # Maybe we should switch to a shared structured format for errors
                # so we can share code with normal errors better.
                failure = str(unused_object)
                print(unused_object)
                all_failures.append(
                    {
                        "filename": node_visitor.UNUSED_OBJECT_FILENAME,
                        "absolute_filename": node_visitor.UNUSED_OBJECT_FILENAME,
                        "message": failure + "\n",
                        "description": failure,
                    }
                )
        if attribute_checker is not None:
            all_failures += attribute_checker.all_failures
        return all_failures

    @classmethod
    def check_file_in_worker(
        cls,
        filename: str,
        attribute_checker: Optional[ClassAttributeChecker] = None,
        **kwargs: Any,
    ) -> Tuple[List[node_visitor.Failure], Any]:
        failures = cls.check_file(
            filename, attribute_checker=attribute_checker, **kwargs
        )
        return failures, attribute_checker

    @classmethod
    def merge_extra_data(
        cls,
        extra_data: Any,
        attribute_checker: Optional[ClassAttributeChecker] = None,
        **kwargs: Any,
    ) -> None:
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

    # Protocol compliance
    def visit_expression(self, node: ast.AST) -> Value:
        return self.visit(node)


def build_stacked_scopes(
    module: Optional[types.ModuleType], simplification_limit: Optional[int] = None
) -> StackedScopes:
    # Build a StackedScopes object.
    # Not part of stacked_scopes.py to avoid a circular dependency.
    if module is None:
        module_vars = {"__name__": TypedValue(str), "__file__": TypedValue(str)}
    else:
        module_vars = {}
        annotations = getattr(module, "__annotations__", {})
        for key, value in module.__dict__.items():
            val = type_from_annotations(annotations, key, globals=module.__dict__)
            if val is None:
                val = KnownValue(value)
            module_vars[key] = val
    return StackedScopes(module_vars, module, simplification_limit=simplification_limit)


def _get_task_cls(fn: object) -> Type[asynq.FutureBase]:
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


def _all_names_unused(
    elts: Iterable[ast.AST], unused_name_nodes: Container[ast.AST]
) -> bool:
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


def _contains_node(elts: Iterable[ast.AST], node: ast.AST) -> bool:
    """Given a list of assignment targets (elts), return whether it contains the given Name node.
    """
    for elt in elts:
        if isinstance(elt, (ast.List, ast.Tuple)):
            if _contains_node(elt.elts, node):
                return True
        if elt is node:
            return True
    return False


def _static_hasattr(value: object, attr: str) -> bool:
    """Returns whether this value has the given attribute, ignoring __getattr__ overrides.
    """
    try:
        object.__getattribute__(value, attr)
    except AttributeError:
        return False
    else:
        return True


def _has_annotation_for_attr(typ: type, attr: str) -> bool:
    try:
        return attr in typ.__annotations__
    except Exception:
        # __annotations__ doesn't exist or isn't a dict
        return False


def _is_asynq_future(value: Value) -> bool:
    return value.is_type(asynq.FutureBase) or value.is_type(asynq.AsyncTask)
