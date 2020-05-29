from __future__ import absolute_import, division, print_function

"""

Implementation of extended argument specifications used by test_scope.

"""

from .annotations import type_from_runtime
from .config import Config
from .error_code import ErrorCode
from .format_strings import parse_format_string
from .stacked_scopes import NULL_CONSTRAINT, Constraint, ConstraintType, OrConstraint
from .value import (
    TypedValue,
    SubclassValue,
    GenericValue,
    DictIncompleteValue,
    SequenceIncompleteValue,
    KnownValue,
    MultiValuedValue,
    UNRESOLVED_VALUE,
    Value,
    VariableNameValue,
    AwaitableIncompleteValue,
)

import ast

try:
    import asyncio
except ImportError:
    asyncio = None
import asynq
from six.moves import builtins, reduce
import collections
import contextlib
import qcore
from qcore import InspectableClass
import inspect
import logging
import re
import six
import sys
import warnings

try:
    import typeshed_client
except ImportError:
    typeshed_client = None


NON_IDENTIFIER_CHARS = re.compile(r"[^a-zA-Z_\d]")
_NO_ARG_SENTINEL = qcore.MarkerObject("no argument given")
_NO_VALUE = qcore.MarkerObject("no value")


def assert_is_value(obj, value):
    """Used to test test_scope's value inference.

    Takes two arguments: a Python object and a Value object. This function does nothing at runtime,
    but test_scope checks that when it encounters a call to assert_is_value, the inferred value of
    the object matches that in the call.

    """
    pass


def dump_value(value):
    """Used for debugging test_scope.

    Calling it will make test_scope print out the argument's inferred value. Does nothing at
    runtime.

    """
    pass


@contextlib.contextmanager
def with_implementation(fn, implementation_fn):
    """Temporarily sets the implementation of fn to be implementation_fn.

    This is useful for invoking test_scope to aggregate all calls to a particular function. For
    example, the following can be used to find the names of all scribe categories we log to:

        categories = set()
        def _scribe_log_impl(variables, visitor, node):
            if isinstance(variables['category'], pyanalyze.value.KnownValue):
                categories.add(variables['category'].val)

        with pyanalyze.arg_spec.with_implementation(qclient.scribe.log, _scribe_log_impl):
            test_scope.test_all()

        print(categories)

    """
    if fn in ArgSpecCache.DEFAULT_ARGSPECS:
        with qcore.override(
            ArgSpecCache.DEFAULT_ARGSPECS[fn], "implementation", implementation_fn
        ):
            yield
    else:
        argspec = ArgSpecCache(Config()).get_argspec(
            fn, implementation=implementation_fn
        )
        if argspec is None:
            # builtin or something, just use a generic argspec
            argspec = ExtendedArgSpec(
                [], starargs="args", kwargs="kwargs", implementation=implementation_fn
            )
        known_argspecs = dict(ArgSpecCache.DEFAULT_ARGSPECS)
        known_argspecs[fn] = argspec
        with qcore.override(ArgSpecCache, "DEFAULT_ARGSPECS", known_argspecs):
            yield


class Parameter(InspectableClass):
    """Class representing a function parameter.

    default_value is Parameter.no_default_value when there is no default value.

    """

    no_default_value = object()

    def __init__(self, name, default_value=no_default_value, typ=None):
        self.name = name
        self.default_value = default_value
        if isinstance(typ, (type, tuple)):
            typ = TypedValue(typ)
        self.typ = typ


class BoundMethodArgSpecWrapper(InspectableClass):
    """Wrapper around ExtendedArgSpec to support bound methods.

    Adds the object that the method is bound to as an argument.

    """

    def __init__(self, argspec, self_value):
        assert isinstance(argspec, ExtendedArgSpec), "invalid argspec %r" % (argspec,)
        self.argspec = argspec
        self.self_value = self_value

    def check_call(self, args, keywords, visitor, node):
        return self.argspec.check_call(
            [self.self_value] + args, keywords, visitor, node
        )

    def has_return_value(self):
        return self.argspec.has_return_value()

    @property
    def return_value(self):
        return self.argspec.return_value


class PropertyArgSpec(InspectableClass):
    """Pseudo-argspec for properties."""

    def __init__(self, obj, return_value=UNRESOLVED_VALUE):
        self.obj = obj
        self._has_return_value = return_value is not UNRESOLVED_VALUE
        self.return_value = return_value

    def check_call(self, args, keywords, visitor, node):
        raise TypeError("property object is not callable")

    def has_return_value(self):
        return self._has_return_value


class ExtendedArgSpec(InspectableClass):
    """A richer version of the standard inspect.ArgSpec object.

    This stores:
    - arguments (list of Parameters): all standard arguments taken by the object.
    - starargs (string or None): name of the *args argument, if present.
    - kwargs (string or None): name of the **kwargs argument, if present.
    - kwonly_args (list of Parameters): list of keyword-only arguments taken by the object. Python 2
      itself does not support this, but some of our objects simulate keyword-only arguments by
      working with **kwargs.
    - return_value (Value): what the object returns
    - name (string): name of the object. This is used only for error output.
    - implementation (function): a function that implements the object. This gets passed a
      dictionary corresponding to the function's scope, a NameCheckVisitor object, and the node
      where the function was called. This can be used for more complicated argument checking and
      for computing the return value from the arguments.

    """

    _default_value = object()
    _kwonly_args_name = "__kwargs"
    _excluded_attributes = {"logger"}

    def __init__(
        self,
        arguments,
        starargs=None,
        kwargs=None,
        kwonly_args=None,
        return_value=UNRESOLVED_VALUE,
        name=None,
        logger=None,
        implementation=None,
    ):
        self.arguments = arguments
        self.starargs = starargs
        self.kwargs = kwargs
        if kwonly_args == []:
            kwonly_args = None
        self.kwonly_args = kwonly_args
        self._has_return_value = return_value is not UNRESOLVED_VALUE
        self.return_value = return_value
        self.name = name
        self.logger = logger
        self.implementation = implementation

        self.params_of_names = {}
        for param in arguments:
            self.params_of_names[param.name] = param
        if kwonly_args is not None:
            for param in kwonly_args:
                self.params_of_names[param.name] = param
        if starargs is not None:
            self.params_of_names[starargs] = Parameter(starargs, typ=tuple)
        if kwargs is not None:
            self.params_of_names[kwargs] = Parameter(kwargs, typ=dict)

    def log(self, level, label, value):
        if self.logger is not None:
            self.logger(level, label, value)

    @qcore.caching.cached_per_instance()
    def generate_function(self):
        """Generates a function with this argspec.

        This is done by exec-ing code that corresponds to this argspec. The function will return
        its locals(). Keyword-only arguments are not checked.

        """
        argument_strings = []
        scope = {
            "_default_value": self._default_value,
            # so that we can call locals() from inside the function even if a function argument is
            # called locals
            # if you want to call a function argument __builtin__, go do it somewhere else please
            "__builtin__": builtins,
        }

        def add_arg(arg):
            if arg.default_value is Parameter.no_default_value:
                argument_strings.append(arg.name)
            else:
                default_obj_name = "__default_" + arg.name
                scope[default_obj_name] = KnownValue(arg.default_value)
                argument_strings.append("%s=%s" % (arg.name, default_obj_name))

        for arg in self.arguments:
            add_arg(arg)

        if self.starargs is not None:
            argument_strings.append("*%s" % self.starargs)

        if six.PY2:
            if self.kwargs is not None:
                argument_strings.append("**%s" % self.kwargs)
            elif self.kwonly_args is not None:
                argument_strings.append("**%s" % self._kwonly_args_name)
        else:
            if self.kwonly_args is not None:
                if self.starargs is None:
                    argument_strings.append("*")
                for arg in self.kwonly_args:
                    add_arg(arg)
            if self.kwargs is not None:
                argument_strings.append("**%s" % self.kwargs)

        if self.name is None:
            name = "test_function"
        else:
            name = six.text_type(self.name)
        # for lambdas name is "<lambda>"
        name = NON_IDENTIFIER_CHARS.sub("_", name)

        code_str = """
def %(name)s(%(arguments)s):
    return __builtin__.locals()
""" % {
            "arguments": ", ".join(argument_strings),
            "name": name,
        }

        self.log(logging.DEBUG, "Code to execute", code_str)
        exec (code_str, scope)
        return scope[name]

    def _check_param_type_compatibility(self, param, var_value, visitor, node):
        if param.typ is not None and var_value != KnownValue(param.default_value):
            compatible = visitor.is_value_compatible(param.typ, var_value)
            if not compatible:
                visitor.show_error(
                    node,
                    "Incompatible argument type for %s: expected %s but got %s"
                    % (param.name, param.typ, var_value),
                    ErrorCode.incompatible_argument,
                )

    def check_call(self, args, keywords, visitor, node):
        """Tries to call this object with the given arguments and keyword arguments.

        Raises a TypeError if something goes wrong.

        This is done by calling the function generated by generate_function(), and then examining
        the local variables to validate types and keyword-only arguments.

        """
        fn = self.generate_function()
        try:
            variables = fn(*args, **dict(keywords))
        except TypeError as e:
            visitor.show_error(node, repr(e), ErrorCode.incompatible_call)
            return UNRESOLVED_VALUE, NULL_CONSTRAINT, NULL_CONSTRAINT
        self.log(logging.DEBUG, "Variables from function call", variables)
        non_param_names = {self.starargs, self.kwargs, self._kwonly_args_name}
        for name, var_value in six.iteritems(variables):
            if var_value is not self._default_value and name not in non_param_names:
                param = self.params_of_names[name]
                self._check_param_type_compatibility(param, var_value, visitor, node)
        if six.PY2 and self.kwonly_args is not None:
            varname = self.kwargs if self.kwargs is not None else self._kwonly_args_name
            kwargs = variables[varname]
            if self.kwargs is None:
                unexpected_args = set(kwargs.keys()) - {
                    param.name for param in self.kwonly_args
                }
                if len(unexpected_args) > 0:
                    visitor.show_error(
                        node,
                        "Unexpected keyword arguments: %r" % ", ".join(unexpected_args),
                        ErrorCode.incompatible_call,
                    )
                for param in self.kwonly_args:
                    if param.name in kwargs:
                        self._check_param_type_compatibility(
                            param, kwargs[param.name], visitor, node
                        )
                    elif param.default_value is Parameter.no_default_value:
                        visitor.show_error(
                            node,
                            "Required keyword-only argument was not passed in: %r"
                            % param.name,
                            ErrorCode.incompatible_call,
                        )

        if self.implementation is not None:
            self.log(logging.DEBUG, "Using implementation", self.implementation)
            return_value = self.implementation(variables, visitor, node)
            if return_value is None:
                return_value = UNRESOLVED_VALUE
            # Implementation functioons may return a pair of (value, constraint)
            # Or a three-tuple of (value, constraint, NoReturn unless)
            if isinstance(return_value, tuple):
                if len(return_value) == 2:
                    return_value, constraint = return_value
                    no_return_unless = NULL_CONSTRAINT
                elif len(return_value) == 3:
                    return_value, constraint, no_return_unless = return_value
                else:
                    assert False, (
                        "%s implementation must return a 2- or 3-tuple, not %s"
                        % (self, return_value)
                    )
            else:
                constraint = no_return_unless = NULL_CONSTRAINT
            # this indicates a bug in test_scope, so using assert
            assert isinstance(return_value, Value), (
                "%s implementation did not return a Value" % self
            )
            return return_value, constraint, no_return_unless
        else:
            return self.return_value, NULL_CONSTRAINT, NULL_CONSTRAINT

    def has_return_value(self):
        # We can't check self.return_value directly here because that may have
        # been wrapped in AwaitableIncompleteValue.
        return self._has_return_value


def is_dot_asynq_function(obj):
    """Returns whether obj is the .asynq member on an async function."""
    try:
        self_obj = obj.__self__
    except AttributeError:
        # the attribute doesn't exist
        return False
    except Exception:
        # The object has a buggy __getattr__ that threw an error. Just ignore it.
        return False
    if qcore.inspection.is_classmethod(obj):
        return False
    if obj is self_obj:
        return False
    try:
        is_async_fn = asynq.is_async_fn(self_obj)
    except Exception:
        # The object may have a buggy __getattr__. Ignore it. This happens with
        # pylons request objects.
        return False
    if not is_async_fn:
        return False

    return getattr(obj, "__name__", None) in ("async", "asynq")


# Implementations of some important functions for use in their ExtendedArgSpecs (see above). These
# are called when the test_scope checker encounters call to these functions. They are passed
# the following arguments:
# - variables: a dictionary with the function's arguments (keys are variable names and values are
#   Value objects)
# - visitor: the test_scope NameCheckVisitor object, which can be used to check context and emit
#   errors
# - node: the AST node corresponding to the call, which needs to be given in order to show errors.
# They return either a single Value object, indicating what the function returns, or a tuple of two
# or three elements:
# - The return value
# - A Constraint indicating things that are true if the function returns a truthy value
# - A Constraint indicating things that are true unless the function does not return
def _isinstance_impl(variables, visitor, node):
    class_or_tuple = variables["class_or_tuple"]
    if not isinstance(class_or_tuple, KnownValue):
        return TypedValue(bool), NULL_CONSTRAINT
    if len(node.args) < 1:
        return TypedValue(bool), NULL_CONSTRAINT
    varname = visitor.varname_for_constraint(node.args[0])
    if varname is None:
        return TypedValue(bool), NULL_CONSTRAINT
    if isinstance(class_or_tuple.val, type):
        return (
            TypedValue(bool),
            Constraint(varname, ConstraintType.is_instance, True, class_or_tuple.val),
        )
    elif isinstance(class_or_tuple.val, tuple) and all(
        isinstance(elt, type) for elt in class_or_tuple.val
    ):
        constraints = [
            Constraint(varname, ConstraintType.is_instance, True, elt)
            for elt in class_or_tuple.val
        ]
        return TypedValue(bool), reduce(OrConstraint, constraints)
    else:
        return TypedValue(bool), NULL_CONSTRAINT


def _hasattr_impl(variables, visitor, node):
    obj = variables["object"]
    name = variables["name"]
    if not isinstance(name, KnownValue):
        return TypedValue(bool)
    if not isinstance(obj, (TypedValue, KnownValue)):
        return TypedValue(bool)

    typ = obj.typ if isinstance(obj, TypedValue) else type(obj.val)
    # interpret a hasattr check as a sign that the object (somehow) has the attribute
    visitor._record_type_attr_set(typ, name.val, node, UNRESOLVED_VALUE)

    # if the value exists on the type or instance, hasattr should return True
    # don't interpret the opposite to mean it should return False, as the attribute may
    # exist on a child class or get assigned at runtime
    if isinstance(obj, TypedValue) and _safe_has_attr(obj.typ, name.val):
        return KnownValue(True)
    elif isinstance(obj, KnownValue) and _safe_has_attr(obj.val, name.val):
        return KnownValue(True)
    else:
        return TypedValue(bool)


def _safe_has_attr(item, member):
    try:
        # some sketchy implementation (like paste.registry) of
        # __getattr__ caused errors at static analysis.
        return hasattr(item, member)
    except Exception:
        return False


def _setattr_impl(variables, visitor, node):
    # if we set an attribute on a value of known type, record it to the attribute checker so we
    # don't say the attribute is undefined
    obj = variables["object"]
    name = variables["name"]
    if isinstance(obj, TypedValue):
        typ = obj.typ
        if isinstance(name, KnownValue):
            visitor._record_type_attr_set(typ, name.val, node, variables["value"])
        else:
            visitor._record_type_has_dynamic_attrs(typ)
    return KnownValue(None)


def _len_impl(variables, visitor, node):
    obj = variables["object"]
    typ = obj.get_type()
    if (
        typ is not None
        and not hasattr(typ, "__len__")
        and typ not in visitor.config.IGNORED_TYPES
    ):
        visitor.show_error(
            node,
            "object of type %s has no len()" % (typ,),
            error_code=ErrorCode.incompatible_argument,
        )
    return TypedValue(int)


def _super_impl(variables, visitor, node):
    typ = variables["type"]
    obj = variables["obj"]
    if six.PY3 and typ == KnownValue(None):
        # Zero-argument super()
        if visitor.in_comprehension_body:
            visitor.show_error(
                node,
                "Zero-argument super() does not work inside a comprehension",
                ErrorCode.bad_super_call,
            )
        elif visitor.scope.is_nested_function():
            visitor.show_error(
                node,
                "Zero-argument super() does not work inside a nested function",
                ErrorCode.bad_super_call,
            )
        current_class = visitor.asynq_checker.current_class
        if current_class is not None:
            try:
                first_arg = visitor.scope.current_scope().get(
                    "%first_arg", None, visitor.state
                )
            except KeyError:
                # something weird with this function; give up
                visitor.show_error(
                    node, "failed to find %first_arg", ErrorCode.bad_super_call
                )
                return UNRESOLVED_VALUE
            else:
                if isinstance(first_arg, SubclassValue):
                    return KnownValue(super(current_class, first_arg.typ))
                elif isinstance(first_arg, KnownValue):
                    return KnownValue(super(current_class, first_arg.val))
                elif isinstance(first_arg, TypedValue):
                    return TypedValue(super(current_class, first_arg.typ))
                else:
                    return UNRESOLVED_VALUE
        return UNRESOLVED_VALUE

    if isinstance(typ, KnownValue):
        if inspect.isclass(typ.val):
            cls = typ.val
        else:
            visitor.show_error(
                node,
                "First argument to super must be a class",
                ErrorCode.bad_super_call,
            )
            return UNRESOLVED_VALUE
    else:
        return UNRESOLVED_VALUE  # probably a dynamically created class

    if isinstance(obj, TypedValue) and obj.typ is not type:
        instance_type = obj.typ
        is_value = True
    elif isinstance(obj, SubclassValue):
        instance_type = obj.typ
        is_value = False
    else:
        return UNRESOLVED_VALUE

    if not issubclass(instance_type, cls):
        visitor.show_error(
            node, "Incompatible arguments to super", ErrorCode.bad_super_call
        )

    current_class = visitor.asynq_checker.current_class
    if current_class is not None and cls is not current_class:
        visitor.show_error(
            node,
            "First argument to super() is not the current class",
            ErrorCode.bad_super_call,
        )

    try:
        super_val = super(cls, instance_type)
    except Exception:
        visitor.show_error(node, "Bad arguments to super", ErrorCode.bad_super_call)
        return UNRESOLVED_VALUE

    if is_value:
        return TypedValue(super_val)
    else:
        return KnownValue(super_val)


def _tuple_impl(variables, visitor, node):
    return _sequence_impl(tuple, variables, visitor, node)


def _list_impl(variables, visitor, node):
    return _sequence_impl(list, variables, visitor, node)


def _set_impl(variables, visitor, node):
    return _sequence_impl(set, variables, visitor, node)


def _sequence_impl(typ, variables, visitor, node):
    iterable = variables["iterable"]
    if iterable == KnownValue(_NO_ARG_SENTINEL):
        return KnownValue(typ())
    elif isinstance(iterable, KnownValue):
        try:
            return KnownValue(typ(iterable.val))
        except TypeError:
            if iterable.val is not None:
                visitor.show_error(
                    node,
                    "Object %r is not iterable" % (iterable.val,),
                    ErrorCode.unsupported_operation,
                )
            return TypedValue(typ)
    elif isinstance(iterable, SequenceIncompleteValue):
        return SequenceIncompleteValue(typ, iterable.members)
    elif isinstance(iterable, DictIncompleteValue):
        return SequenceIncompleteValue(typ, [key for key, _ in iterable.items])
    elif isinstance(iterable, TypedValue):
        if not iterable.is_type(
            collections.Iterable
        ) and not visitor._should_ignore_type(iterable.typ):
            visitor.show_error(
                node,
                "Object of type %r is not iterable" % (iterable.typ,),
                ErrorCode.unsupported_operation,
            )
        if isinstance(iterable, GenericValue):
            return GenericValue(typ, [iterable.get_arg(0)])
    return TypedValue(typ)


def _assert_is_value_impl(variables, visitor, node):
    if not visitor._is_checking():
        return KnownValue(None)
    obj = variables["obj"]
    expected_value = variables["value"]
    if not isinstance(expected_value, KnownValue):
        visitor.show_error(
            node,
            "Value argument to assert_is_value must be a KnownValue (got %r) %r"
            % (expected_value, obj),
            ErrorCode.inference_failure,
        )
    else:
        if obj != expected_value.val:
            visitor.show_error(
                node,
                "Bad value inference: expected %r, got %r" % (expected_value.val, obj),
                ErrorCode.inference_failure,
            )
    return KnownValue(None)


def _dump_value_impl(variables, visitor, node):
    if visitor._is_checking():
        visitor.show_error(
            node, "value: %r" % variables["value"], ErrorCode.inference_failure
        )
    return KnownValue(None)


def _xrange_impl(variables, visitor, node):
    for name in ("start", "stop", "step"):
        val = variables[name]
        if isinstance(val, KnownValue) and val.val is not None:
            if val.val >= 2 ** 31:
                new_node = ast.Name(id="lrange")
                visitor._maybe_show_missing_import_error(new_node)
                visitor.show_error(
                    node,
                    "xrange does not support arguments greater than 2**31 (got %s)"
                    % val.val,
                    ErrorCode.incompatible_argument,
                    replacement=visitor.replace_node(node.func, new_node),
                )
    return UNRESOLVED_VALUE


def _py2_input_impl(variables, visitor, node):
    visitor.show_error(
        node,
        "Do not use input(); it is unsafe. Use raw_input() instead or use "
        '"from six.moves import input".',
        ErrorCode.incompatible_call,
    )
    return UNRESOLVED_VALUE


def _str_format_impl(variables, visitor, node):
    return _format_impl(six.text_type, variables, visitor, node)


def _bytes_format_impl(variables, visitor, node):
    return _format_impl(bytes, variables, visitor, node)


def _format_impl(typ, variables, visitor, node):
    self = variables["self"]
    if not isinstance(self, KnownValue):
        return TypedValue(typ)
    args = variables["args"]
    kwargs = variables["kwargs"]
    template = self.val
    used_indices = set()
    used_kwargs = set()
    current_index = 0
    parsed, errors = parse_format_string(template)
    if errors:
        _, message = errors[0]
        visitor.show_error(node, message, error_code=ErrorCode.incompatible_call)
        return TypedValue(typ)
    for field in parsed.iter_replacement_fields():
        # TODO validate conversion specifiers, attributes, etc.
        if field.arg_name is None:
            if current_index >= len(args):
                visitor.show_error(
                    node,
                    "Too few arguments to format string (expected at least %s)"
                    % (current_index,),
                    error_code=ErrorCode.incompatible_call,
                )
            used_indices.add(current_index)
            current_index += 1
        elif isinstance(field.arg_name, int):
            index = field.arg_name
            if index >= len(args):
                visitor.show_error(
                    node,
                    "Numbered argument %s to format string is out of range" % (index,),
                    error_code=ErrorCode.incompatible_call,
                )
            used_indices.add(index)
        else:
            if field.arg_name not in kwargs:
                visitor.show_error(
                    node,
                    "Named argument %s to format string was not given"
                    % (field.arg_name,),
                    error_code=ErrorCode.incompatible_call,
                )
            used_kwargs.add(field.arg_name)
    unused_indices = set(range(len(args))) - used_indices
    if unused_indices:
        visitor.show_error(
            node,
            "Numbered argument(s) %s were not used"
            % (", ".join(map(str, sorted(unused_indices)))),
            error_code=ErrorCode.incompatible_call,
        )
    unused_kwargs = set(kwargs) - used_kwargs
    if unused_kwargs:
        visitor.show_error(
            node,
            "Named argument(s) %s were not used" % (", ".join(sorted(unused_kwargs))),
            error_code=ErrorCode.incompatible_call,
        )
    return TypedValue(typ)


def _assert_is_impl(variables, visitor, node):
    return _qcore_assert_impl(variables, visitor, node, ConstraintType.is_value, True)


def _assert_is_not_impl(variables, visitor, node):
    return _qcore_assert_impl(variables, visitor, node, ConstraintType.is_value, False)


def _qcore_assert_impl(variables, visitor, node, constraint_type, positive):
    if len(node.args) < 2:
        # arguments were passed as kwargs
        return KnownValue(None), NULL_CONSTRAINT, NULL_CONSTRAINT
    left_varname = visitor.varname_for_constraint(node.args[0])
    right_varname = visitor.varname_for_constraint(node.args[1])
    if left_varname is not None:
        if not isinstance(variables["actual"], KnownValue):
            return KnownValue(None), NULL_CONSTRAINT, NULL_CONSTRAINT
        varname = left_varname
        constrained_to = variables["actual"].val
    elif right_varname is not None:
        if not isinstance(variables["expected"], KnownValue):
            return KnownValue(None), NULL_CONSTRAINT, NULL_CONSTRAINT
        varname = left_varname
        constrained_to = variables["expected"].val
    else:
        return KnownValue(None), NULL_CONSTRAINT, NULL_CONSTRAINT

    no_return_unless = Constraint(varname, constraint_type, positive, constrained_to)
    return KnownValue(None), NULL_CONSTRAINT, no_return_unless


if six.PY3:
    _ENCODING_PARAMETER = Parameter("encoding", typ=str, default_value="")
else:
    # In Python 2, enforce that encoding is given so that we're more explicit
    # in dealing with str/bytes.
    _ENCODING_PARAMETER = Parameter("encoding", typ=six.string_types)


class ArgSpecCache(object):
    DEFAULT_ARGSPECS = {
        assert_is_value: ExtendedArgSpec(
            [Parameter("obj"), Parameter("value", typ=Value)],
            implementation=_assert_is_value_impl,
            name="assert_is_value",
        ),
        dump_value: ExtendedArgSpec(
            [Parameter("value")], implementation=_dump_value_impl, name="dump_value"
        ),
        # builtins
        isinstance: ExtendedArgSpec(
            [Parameter("obj"), Parameter("class_or_tuple")],
            name="isinstance",
            implementation=_isinstance_impl,
        ),
        getattr: ExtendedArgSpec(
            [
                Parameter("object"),
                Parameter("name", typ=six.string_types),
                Parameter("default", default_value=None),
            ],
            name="getattr",
        ),
        hasattr: ExtendedArgSpec(
            [Parameter("object"), Parameter("name", typ=six.string_types)],
            return_value=TypedValue(bool),
            name="hasattr",
            implementation=_hasattr_impl,
        ),
        setattr: ExtendedArgSpec(
            [
                Parameter("object"),
                Parameter("name", typ=six.string_types),
                Parameter("value"),
            ],
            return_value=KnownValue(None),
            name="setattr",
            implementation=_setattr_impl,
        ),
        super: ExtendedArgSpec(
            [
                Parameter(
                    "type",
                    default_value=None if six.PY3 else Parameter.no_default_value,
                ),
                Parameter("obj", default_value=None),
            ],
            name="super",
            implementation=_super_impl,
        ),
        tuple: ExtendedArgSpec(
            [Parameter("iterable", default_value=_NO_ARG_SENTINEL)],
            name="tuple",
            implementation=_tuple_impl,
        ),
        list: ExtendedArgSpec(
            [Parameter("iterable", default_value=_NO_ARG_SENTINEL)],
            name="list",
            implementation=_list_impl,
        ),
        set: ExtendedArgSpec(
            [Parameter("iterable", default_value=_NO_ARG_SENTINEL)],
            name="set",
            implementation=_set_impl,
        ),
        bytes.decode: ExtendedArgSpec(
            [
                Parameter("self", typ=bytes),
                _ENCODING_PARAMETER,
                Parameter("errors", typ=six.string_types, default_value=""),
            ],
            name="bytes.decode",
            return_value=TypedValue(six.text_type),
        ),
        six.text_type.encode: ExtendedArgSpec(
            [
                Parameter("self", typ=six.text_type),
                _ENCODING_PARAMETER,
                Parameter("errors", typ=six.string_types, default_value=""),
            ],
            name="{}.encode".format(six.text_type.__name__),
            return_value=TypedValue(bytes),
        ),
        six.text_type.format: ExtendedArgSpec(
            [Parameter("self", typ=six.text_type)],
            starargs="args",
            kwargs="kwargs",
            name="{}.format".format(six.text_type.__name__),
            implementation=_str_format_impl,
        ),
        # workaround for https://github.com/python/typeshed/pull/3501
        warnings.warn: ExtendedArgSpec(
            [
                Parameter(
                    "message",
                    typ=MultiValuedValue([TypedValue(str), TypedValue(Warning)]),
                ),
                Parameter("category", typ=UNRESOLVED_VALUE, default_value=None),
                Parameter("stacklevel", typ=int, default_value=1),
            ],
            name="warnings.warn",
            return_value=None,
        ),
        # qcore/asynq
        # just so we can infer the return value
        qcore.utime: ExtendedArgSpec([], name="utime", return_value=TypedValue(int)),
        qcore.asserts.assert_is: ExtendedArgSpec(
            [
                Parameter("expected"),
                Parameter("actual"),
                Parameter("message", default_value=None,),
                Parameter("extra", default_value=None,),
            ],
            name="assert_is",
            implementation=_assert_is_impl,
        ),
        qcore.asserts.assert_is_not: ExtendedArgSpec(
            [
                Parameter("expected"),
                Parameter("actual"),
                Parameter("message", default_value=None,),
                Parameter("extra", default_value=None,),
            ],
            name="assert_is_not",
            implementation=_assert_is_not_impl,
        ),
    }
    if six.PY2:
        DEFAULT_ARGSPECS[str.format] = ExtendedArgSpec(
            [Parameter("self", typ=str)],
            starargs="args",
            kwargs="kwargs",
            name="str.format",
            implementation=_bytes_format_impl,
        )
        # static analysis: ignore[undefined_name]
        DEFAULT_ARGSPECS[xrange] = ExtendedArgSpec(
            [
                Parameter("start", typ=six.integer_types),
                Parameter("stop", typ=six.integer_types, default_value=None),
                Parameter("step", typ=six.integer_types, default_value=None),
            ],
            name="xrange",
            implementation=_xrange_impl,
        )
        DEFAULT_ARGSPECS[input] = ExtendedArgSpec(
            [Parameter("start", typ=six.string_types, default_value=None)],
            name="input",
            implementation=_py2_input_impl,
        )
    if sys.version_info < (3, 6):
        # Not needed in Python 3.6+ because we have typeshed
        DEFAULT_ARGSPECS[len] = ExtendedArgSpec(
            [Parameter("object")],
            return_value=TypedValue(int),
            name="len",
            implementation=_len_impl,
        )

    def __init__(self, config):
        self.config = config
        self.ts_finder = TypeshedFinder(verbose=False)
        self.known_argspecs = {}
        default_argspecs = dict(self.DEFAULT_ARGSPECS)
        default_argspecs.update(self.config.get_known_argspecs(self))

        for obj, argspec in six.iteritems(default_argspecs):
            # unbound methods in py2
            obj = getattr(obj, "__func__", obj)
            self.known_argspecs[obj] = argspec

    def __reduce_ex__(self, proto):
        # Don't pickle the actual argspecs, which are frequently unpicklable.
        return self.__class__, (self.config,)

    def from_argspec(
        self,
        argspec,
        kwonly_args=None,
        name=None,
        logger=None,
        implementation=None,
        function_object=None,
    ):
        """Constructs an ExtendedArgSpec from a standard argspec.

        argspec can be either an inspect.ArgSpec or, in Python 3 only, an inspect.FullArgSpec, with
        support for keyword-only arguments, or inspect.Signature.

        kwonly_args may be a list of custom keyword-only arguments added to the argspec or None.

        name is the name of the function. This is used for better error messages.

        logger is the log function to be used.

        implementation is an implementation function for this object.

        """
        if argspec is None:
            return None
        if kwonly_args is None:
            kwonly_args = []
        else:
            kwonly_args = list(kwonly_args)
        func_globals = getattr(function_object, "__globals__", None)

        if hasattr(argspec, "parameters"):
            # inspect.Signature object
            starargs = None
            kwargs = None
            args = []
            if argspec.return_annotation is argspec.empty:
                return_value = UNRESOLVED_VALUE
            else:
                return_value = type_from_runtime(
                    argspec.return_annotation, globals=func_globals
                )
            for parameter in argspec.parameters.values():
                if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                    starargs = parameter.name
                elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                    kwargs = parameter.name
                elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                    kwonly_args.append(
                        self._parameter_from_signature(parameter, func_globals)
                    )
                else:
                    # positional or positional-or-keyword
                    args.append(self._parameter_from_signature(parameter, func_globals))
        else:
            if argspec.defaults is None:
                arg_pairs = [(arg, Parameter.no_default_value) for arg in argspec.args]
            else:
                num_non_default = len(argspec.args) - len(argspec.defaults)
                defaults = [Parameter.no_default_value] * num_non_default
                for default in argspec.defaults:
                    # hack to enable test_scope to run on itself
                    if default is Parameter.no_default_value:
                        defaults.append(object())
                    else:
                        defaults.append(default)
                arg_pairs = zip(argspec.args, defaults)

            args = []
            for arg, default_value in arg_pairs:
                args.append(
                    Parameter(
                        arg,
                        default_value=default_value,
                        typ=VariableNameValue.from_varname(
                            arg, self.config.varname_value_map()
                        ),
                    )
                )

            # python 3 keyword-only arguments
            if hasattr(argspec, "kwonlyargs") and argspec.kwonlyargs:
                kwonlydefaults = (
                    argspec.kwonlydefaults if argspec.kwonlydefaults is not None else {}
                )

                for arg in argspec.kwonlyargs:
                    kwonly_args.append(
                        Parameter(
                            arg,
                            default_value=kwonlydefaults.get(
                                arg, Parameter.no_default_value
                            ),
                        )
                    )

            # FullArgSpec has varkw and ArgSpec has keywords
            kwargs = argspec.keywords if hasattr(argspec, "keywords") else argspec.varkw
            return_value = UNRESOLVED_VALUE
            starargs = argspec.varargs
        return ExtendedArgSpec(
            args,
            starargs=starargs,
            kwargs=kwargs,
            kwonly_args=kwonly_args or None,
            return_value=return_value,
            name=name,
            logger=logger,
            implementation=implementation,
        )

    def _parameter_from_signature(self, parameter, func_globals):
        """Given an inspect.Parameter, returns a Parameter object."""
        if parameter.annotation is not inspect.Parameter.empty:
            typ = type_from_runtime(parameter.annotation, globals=func_globals)
        else:
            typ = VariableNameValue.from_varname(
                parameter.name, self.config.varname_value_map()
            )
        if parameter.default is inspect.Parameter.empty:
            default_value = Parameter.no_default_value
        elif parameter.default is Parameter.no_default_value:
            # hack to prevent errors in code that use Parameter
            default_value = object()
        else:
            default_value = parameter.default
        return Parameter(parameter.name, default_value=default_value, typ=typ)

    def get_argspec(self, obj, name=None, logger=None, implementation=None):
        """Constructs the ExtendedArgSpec for a Python object."""
        kwargs = {"name": name, "logger": logger, "implementation": implementation}
        argspec = self._cached_get_argspec(obj, kwargs)
        return argspec

    def _cached_get_argspec(self, obj, kwargs):
        try:
            if obj in self.known_argspecs:
                return self.known_argspecs[obj]
        except Exception:
            hashable = False  # unhashable, or __eq__ failed
        else:
            hashable = True

        extended = self._uncached_get_argspec(obj, kwargs)
        if extended is None:
            return None

        if hashable:
            self.known_argspecs[obj] = extended
        return extended

    def _uncached_get_argspec(self, obj, kwargs):
        if isinstance(obj, tuple) or hasattr(obj, "__getattr__"):
            return None  # lost cause

        # Cythonized methods, e.g. fn.asynq
        if is_dot_asynq_function(obj):
            try:
                return self._cached_get_argspec(obj.__self__, kwargs)
            except TypeError:
                # some cythonized methods have __self__ but it is not a function
                pass

        # for bound methods, see if we have an argspec for the unbound method
        if inspect.ismethod(obj) and obj.__self__ is not None:
            argspec = self._cached_get_argspec(obj.__func__, kwargs)
            if argspec is None:
                return None
            return BoundMethodArgSpecWrapper(argspec, KnownValue(obj.__self__))

        if hasattr(obj, "fn") or hasattr(obj, "original_fn"):
            # many decorators put the original function in the .fn attribute
            try:
                original_fn = qcore.get_original_fn(obj)
            except (TypeError, AttributeError):
                # fails when executed on an object that doesn't allow setting attributes,
                # e.g. certain extension classes
                pass
            else:
                return self._cached_get_argspec(original_fn, kwargs)

        argspec = self.ts_finder.get_argspec(obj)
        if argspec is not None:
            if _is_coroutine_function(obj):
                argspec.return_value = AwaitableIncompleteValue(argspec.return_value)
            return argspec

        if inspect.isfunction(obj):
            if hasattr(obj, "inner"):
                # @qclient.task_queue.exec_after_request() puts the original function in .inner
                return self._cached_get_argspec(obj.inner, kwargs)

            argspec = self.from_argspec(
                self._safe_get_argspec(obj), function_object=obj, **kwargs
            )
            if _is_coroutine_function(obj):
                argspec.return_value = AwaitableIncompleteValue(argspec.return_value)
            return argspec

        # decorator binders
        if _is_qcore_decorator(obj):
            argspec = self._cached_get_argspec(obj.decorator, kwargs)
            # wrap if it's a bound method
            if obj.instance is not None and argspec is not None:
                return BoundMethodArgSpecWrapper(argspec, KnownValue(obj.instance))
            return argspec

        if inspect.isclass(obj):
            obj = self.config.unwrap_cls(obj)
            if issubclass(obj, self.config.CLASSES_USING_INIT):
                constructor = obj.init
            elif hasattr(obj, "__init__"):
                constructor = obj.__init__
            else:
                # old-style class
                return None
            argspec = self._safe_get_argspec(constructor)

            kwonly_args = []
            for cls_, args in six.iteritems(
                self.config.CLASS_TO_KEYWORD_ONLY_ARGUMENTS
            ):
                if issubclass(obj, cls_):
                    kwonly_args += [
                        Parameter(param_name, default_value=None) for param_name in args
                    ]
            argspec = self.from_argspec(
                argspec, function_object=constructor, kwonly_args=kwonly_args, **kwargs
            )
            if argspec is None:
                return None
            return BoundMethodArgSpecWrapper(argspec, TypedValue(obj))

        if inspect.isbuiltin(obj):
            if hasattr(obj, "__self__"):
                cls = type(obj.__self__)
                try:
                    method = getattr(cls, obj.__name__)
                except AttributeError:
                    return None
                if method == obj:
                    return None
                argspec = self._cached_get_argspec(method, kwargs)
                if argspec is None:
                    return None
                return BoundMethodArgSpecWrapper(argspec, KnownValue(obj.__self__))
            return None

        if hasattr(obj, "__call__"):
            # we could get an argspec here in some cases, but it's impossible to figure out
            # the argspec for some builtin methods (e.g., dict.__init__), and no way to detect
            # these with inspect, so just give up.
            return None

        if isinstance(obj, property):
            # If we know the getter, inherit its return value.
            if obj.fget:
                fget_argspec = self._cached_get_argspec(obj.fget, kwargs)
                if fget_argspec is not None and fget_argspec.has_return_value():
                    return PropertyArgSpec(obj, return_value=fget_argspec.return_value)
            return PropertyArgSpec(obj)

        raise TypeError("%r object is not callable" % (obj,))

    def _safe_get_argspec(self, obj):
        """Wrapper around inspect.getargspec that catches TypeErrors."""
        try:
            # follow_wrapped=True leads to problems with decorators that
            # mess with the arguments, such as mock.patch.
            sig = inspect.signature(obj, follow_wrapped=False)
        except (TypeError, ValueError, AttributeError):
            # TypeError if signature() does not support the object, ValueError
            # if it cannot provide a signature, and AttributeError if we're on
            # Python 2.
            pass
        else:
            # Signature preserves the return annotation for wrapped functions,
            # because @functools.wraps copies the __annotations__ of the wrapped function. We
            # don't want that, because the wrapper may have changed the return type.
            # This caused problems with @contextlib.contextmanager.
            if _safe_has_attr(obj, "__wrapped__"):
                return sig.replace(return_annotation=inspect.Signature.empty)
            else:
                return sig
        try:
            if hasattr(inspect, "getfullargspec"):
                try:
                    return inspect.getfullargspec(obj)  # static analysis: ignore
                except TypeError:
                    # fall back to qcore.inspection
                    return qcore.inspection.getargspec(obj)
            return qcore.inspection.getargspec(obj)
        except TypeError:
            # probably a builtin or Cythonized object
            return None


if typeshed_client is None:
    # Fallback: always fails
    class TypeshedFinder(object):
        def __init__(self, verbose=False):
            pass

        def get_argspec(self, obj):
            return None


else:
    from typed_ast import ast3

    class TypeshedFinder(object):
        def __init__(self, verbose=False):
            self.verbose = verbose
            self.resolver = typeshed_client.Resolver(version=sys.version_info[:2])

        def log(self, message, obj):
            if not self.verbose:
                return
            print("%s: %r" % (message, obj))

        def get_argspec(self, obj):
            if inspect.ismethoddescriptor(obj) and hasattr(obj, "__objclass__"):
                objclass = obj.__objclass__
                fq_name = self._get_fq_name(objclass)
                if fq_name is None:
                    return None
                info = self._get_info_for_name(fq_name)
                argspec = self._get_method_argspec_from_info(
                    info, obj, fq_name, objclass.__module__
                )
                if argspec is not None:
                    self.log("Found argspec", (obj, argspec))
                return argspec

            if inspect.ismethod(obj):
                self.log("Ignoring method", obj)
                return None
            fq_name = self._get_fq_name(obj)
            if fq_name is None:
                return None
            info = self._get_info_for_name(fq_name)
            argspec = self._get_argspec_from_info(info, obj, fq_name, obj.__module__)
            if argspec is not None:
                self.log("Found argspec", (fq_name, argspec))
            return argspec

        def _get_method_argspec_from_info(self, info, obj, fq_name, mod):
            if info is None:
                return None
            elif isinstance(info, typeshed_client.ImportedInfo):
                return self._get_method_argspec_from_info(info.info, obj, fq_name, mod)
            elif isinstance(info, typeshed_client.NameInfo):
                # Note that this doesn't handle names inherited from base classes
                if obj.__name__ in info.child_nodes:
                    child_info = info.child_nodes[obj.__name__]
                    return self._get_argspec_from_info(child_info, obj, fq_name, mod)
                else:
                    return None
            else:
                self.log("Ignoring unrecognized info", (fq_name, info))
                return None

        def _get_fq_name(self, obj):
            try:
                module = obj.__module__
                if module is None:
                    module = "builtins"
                return ".".join([module, obj.__qualname__])
            except (AttributeError, TypeError):
                self.log("Ignoring object without module or qualname", obj)
                return None

        def _get_argspec_from_info(self, info, obj, fq_name, mod):
            if isinstance(info, typeshed_client.NameInfo):
                if isinstance(info.ast, (ast3.FunctionDef, ast3.AsyncFunctionDef)):
                    return self._get_argspec_from_func_def(info.ast, obj, mod)
                else:
                    self.log("Ignoring unrecognized AST", (fq_name, info))
                    return None
            elif isinstance(info, typeshed_client.ImportedInfo):
                return self._get_argspec_from_info(info.info, obj, fq_name, mod)
            elif info is None:
                return None
            else:
                self.log("Ignoring unrecognized info", (fq_name, info))
                return None

        @qcore.caching.cached_per_instance()
        def _get_info_for_name(self, fq_name):
            return self.resolver.get_fully_qualified_name(fq_name)

        def _get_argspec_from_func_def(self, node, obj, mod):
            if node.decorator_list:
                # might be @overload or something else we don't recognize
                return None
            if node.returns is None:
                return_value = UNRESOLVED_VALUE
            else:
                return_value = self._parse_expr(node.returns, mod)
            args = node.args
            if args.vararg is None:
                starargs = None
            else:
                starargs = args.vararg.arg
            if args.kwarg is None:
                kwargs = None
            else:
                kwargs = args.kwarg.arg
            num_without_defaults = len(args.args) - len(args.defaults)
            defaults = [None] * num_without_defaults + args.defaults
            arguments = list(self._parse_param_list(args.args, defaults, mod))
            kwonly = list(
                self._parse_param_list(args.kwonlyargs, args.kw_defaults, mod)
            )
            return ExtendedArgSpec(
                arguments,
                starargs=starargs,
                kwargs=kwargs,
                kwonly_args=kwonly,
                return_value=return_value,
                name=obj.__name__,
            )

        def _parse_param_list(self, args, defaults, module):
            for arg, default in zip(args, defaults):
                typ = None
                if arg.annotation is not None:
                    typ = self._parse_expr(arg.annotation, module)

                if default is None:
                    yield Parameter(arg.arg, typ=typ)
                else:
                    # doesn't matter what the default is
                    yield Parameter(arg.arg, typ=typ, default_value=None)

        def _parse_expr(self, node, module):
            raw = self._parse_ast_node(node, module)
            if raw is _NO_VALUE:
                return UNRESOLVED_VALUE
            else:
                return type_from_runtime(raw)

        def _parse_ast_node(self, node, module):
            if isinstance(node, ast3.NameConstant):
                return node.value
            elif isinstance(node, ast3.Name):
                info = self.resolver.get_fully_qualified_name(
                    "%s.%s" % (module, node.id)
                )
                if info is not None:
                    return self._value_from_info(info, module)
                elif hasattr(builtins, node.id):
                    val = getattr(builtins, node.id)
                    if val is None or isinstance(val, type):
                        return val
            elif isinstance(node, ast3.Subscript):
                value = self._parse_ast_node(node.value, module)
                subscript = self._parse_ast_node(node.slice, module)
                try:
                    return value[subscript]
                except Exception:
                    self.log("Ignoring subscript failure", (value, subscript))
                    return _NO_VALUE
            elif isinstance(node, ast3.Index):
                return self._parse_ast_node(node.value, module)
            elif isinstance(node, ast3.Tuple):
                return tuple(self._parse_ast_node(elt, module) for elt in node.elts)
            elif isinstance(node, ast3.List):
                return [self._parse_ast_node(elt, module) for elt in node.elts]
            elif isinstance(node, ast3.Ellipsis):
                return Ellipsis
            elif isinstance(node, ast3.Attribute):
                value = self._parse_ast_node(node.value, module)
                try:
                    return getattr(value, node.attr)
                except Exception:
                    self.log("Ignoring getattr failure", (value, node.attr))
                    return _NO_VALUE
            self.log("Ignoring node", (node, module))
            return _NO_VALUE

        def _value_from_info(self, info, module):
            if isinstance(info, typeshed_client.ImportedInfo):
                return self._value_from_info(info.info, ".".join(info.source_module))
            elif isinstance(info, typeshed_client.NameInfo):
                try:
                    mod = __import__(module)
                    return getattr(mod, info.name)
                except Exception:
                    self.log("Unable to import", (module, info))
                    return _NO_VALUE
            else:
                self.log("Ignoring info", info)
                return _NO_VALUE


def _is_coroutine_function(obj):
    if asyncio is not None:
        # Python 3.4+
        return asyncio.iscoroutinefunction(obj)
    else:
        # no coroutines for you
        return False


def _is_qcore_decorator(obj):
    try:
        return (
            hasattr(obj, "is_decorator")
            and obj.is_decorator()
            and hasattr(obj, "decorator")
        )
    except Exception:
        # black.Line has an is_decorator attribute but it is not a method
        return False
