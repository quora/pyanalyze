"""

Implementation of extended argument specifications used by test_scope.

"""

from .annotations import (
    type_from_runtime,
    Context,
    type_from_ast,
    is_typing_name,
    type_from_maybe_generic,
)
from .config import Config
from .error_code import ErrorCode
from .find_unused import used
from .format_strings import parse_format_string
from .stacked_scopes import (
    NULL_CONSTRAINT,
    Constraint,
    ConstraintType,
    OrConstraint,
    uniq_chain,
)
from .signature import (
    ExtendedArgSpec,
    Parameter,
    ImplementationFnReturn,
    VarsDict,
    Logger,
    ImplementationFn,
    BoundMethodArgSpecWrapper,
    MaybeArgspec,
    PropertyArgSpec,
)
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
    TypeVarValue,
    extract_typevars,
    substitute_typevars,
)

import asyncio
import ast
import asynq
import builtins
from collections.abc import Awaitable, Collection, Set as AbstractSet, Sized
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import reduce
import collections.abc
import contextlib
import qcore
import inspect
import sys
import warnings
from types import GeneratorType
from typing import (
    cast,
    Any,
    Sequence,
    NewType,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Union,
    Callable,
    Dict,
    List,
    TypeVar,
    TYPE_CHECKING,
)
from typing_extensions import Protocol
import typing_inspect
import typeshed_client
from typed_ast import ast3

if TYPE_CHECKING:
    from .name_check_visitor import NameCheckVisitor

T_co = TypeVar("T_co", covariant=True)


IS_PRE_38 = sys.version_info < (3, 8)
_NO_ARG_SENTINEL = qcore.MarkerObject("no argument given")


@used  # exposed as an API
def assert_is_value(obj: object, value: Value) -> None:
    """Used to test test_scope's value inference.

    Takes two arguments: a Python object and a Value object. This function does nothing at runtime,
    but test_scope checks that when it encounters a call to assert_is_value, the inferred value of
    the object matches that in the call.

    """
    pass


@used  # exposed as an API
def dump_value(value: object) -> None:
    """Used for debugging test_scope.

    Calling it will make test_scope print out the argument's inferred value. Does nothing at
    runtime.

    """
    pass


@used  # exposed as an API
@contextlib.contextmanager
def with_implementation(
    fn: object, implementation_fn: ImplementationFn
) -> Iterable[None]:
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


def is_dot_asynq_function(obj: Any) -> bool:
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
# are called when the test_scope checker encounters call to these functions.
def _isinstance_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
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


def _hasattr_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
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


def _safe_has_attr(item: object, member: str) -> bool:
    try:
        # some sketchy implementation (like paste.registry) of
        # __getattr__ caused errors at static analysis.
        return hasattr(item, member)
    except Exception:
        return False


def _setattr_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
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


def _super_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    typ = variables["type"]
    obj = variables["obj"]
    if typ == KnownValue(None):
        # Zero-argument super()
        if visitor.in_comprehension_body:
            visitor.show_error(
                node,
                "Zero-argument super() does not work inside a comprehension",
                ErrorCode.bad_super_call,
            )
        elif visitor.scopes.is_nested_function():
            visitor.show_error(
                node,
                "Zero-argument super() does not work inside a nested function",
                ErrorCode.bad_super_call,
            )
        current_class = visitor.asynq_checker.current_class
        if current_class is not None:
            try:
                first_arg = visitor.scopes.get("%first_arg", None, visitor.state)
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


def _tuple_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    return _sequence_impl(tuple, variables, visitor, node)


def _list_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    return _sequence_impl(list, variables, visitor, node)


def _set_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    return _sequence_impl(set, variables, visitor, node)


def _sequence_impl(
    typ: type, variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
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
            collections.abc.Iterable
        ) and not visitor._should_ignore_type(iterable.typ):
            visitor.show_error(
                node,
                "Object of type %r is not iterable" % (iterable.typ,),
                ErrorCode.unsupported_operation,
            )
        if isinstance(iterable, GenericValue):
            return GenericValue(typ, [iterable.get_arg(0)])
    return TypedValue(typ)


def _assert_is_value_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
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


def _dump_value_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    if visitor._is_checking():
        visitor.show_error(
            node, "value: %r" % variables["value"], ErrorCode.inference_failure
        )
    return KnownValue(None)


def _str_format_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    self = variables["self"]
    if not isinstance(self, KnownValue):
        return TypedValue(str)
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
        return TypedValue(str)
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
    return TypedValue(str)


def _subclasses_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> Value:
    """Overridden because typeshed types make it (T) => List[T] instead."""
    self_obj = variables["self"]
    if isinstance(self_obj, KnownValue) and isinstance(self_obj.val, type):
        return KnownValue(self_obj.val.__subclasses__())
    return GenericValue(list, [TypedValue(type)])


def _assert_is_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    return _qcore_assert_impl(variables, visitor, node, ConstraintType.is_value, True)


def _assert_is_not_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    return _qcore_assert_impl(variables, visitor, node, ConstraintType.is_value, False)


def _qcore_assert_impl(
    variables: VarsDict,
    visitor: "NameCheckVisitor",
    node: ast.AST,
    constraint_type: ConstraintType,
    positive: bool,
) -> ImplementationFnReturn:
    if len(node.args) < 2:
        # arguments were passed as kwargs
        return KnownValue(None), NULL_CONSTRAINT, NULL_CONSTRAINT
    left_varname = visitor.varname_for_constraint(node.args[0])
    right_varname = visitor.varname_for_constraint(node.args[1])
    if left_varname is not None and isinstance(variables["actual"], KnownValue):
        varname = left_varname
        constrained_to = variables["actual"].val
    elif right_varname is not None and isinstance(variables["expected"], KnownValue):
        varname = right_varname
        constrained_to = variables["expected"].val
    else:
        return KnownValue(None), NULL_CONSTRAINT, NULL_CONSTRAINT

    no_return_unless = Constraint(varname, constraint_type, positive, constrained_to)
    return KnownValue(None), NULL_CONSTRAINT, no_return_unless


_ENCODING_PARAMETER = Parameter("encoding", typ=TypedValue(str), default_value="")


class ArgSpecCache:
    DEFAULT_ARGSPECS = {
        assert_is_value: ExtendedArgSpec(
            [Parameter("obj"), Parameter("value", typ=TypedValue(Value))],
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
                Parameter("name", typ=TypedValue(str)),
                Parameter("default", default_value=None),
            ],
            name="getattr",
        ),
        hasattr: ExtendedArgSpec(
            [Parameter("object"), Parameter("name", typ=TypedValue(str))],
            return_value=TypedValue(bool),
            name="hasattr",
            implementation=_hasattr_impl,
        ),
        setattr: ExtendedArgSpec(
            [
                Parameter("object"),
                Parameter("name", typ=TypedValue(str)),
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
                    default_value=None,
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
                Parameter("self", typ=TypedValue(bytes)),
                _ENCODING_PARAMETER,
                Parameter("errors", typ=TypedValue(str), default_value=""),
            ],
            name="bytes.decode",
            return_value=TypedValue(str),
        ),
        str.encode: ExtendedArgSpec(
            [
                Parameter("self", typ=TypedValue(str)),
                _ENCODING_PARAMETER,
                Parameter("errors", typ=TypedValue(str), default_value=""),
            ],
            name="str.encode",
            return_value=TypedValue(bytes),
        ),
        str.format: ExtendedArgSpec(
            [Parameter("self", typ=TypedValue(str))],
            starargs="args",
            kwargs="kwargs",
            name="str.format",
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
                Parameter("stacklevel", typ=TypedValue(int), default_value=1),
            ],
            name="warnings.warn",
            return_value=KnownValue(None),
        ),
        # qcore/asynq
        # just so we can infer the return value
        qcore.utime: ExtendedArgSpec([], name="utime", return_value=TypedValue(int)),
        qcore.asserts.assert_is: ExtendedArgSpec(
            [
                Parameter("expected"),
                Parameter("actual"),
                Parameter("message", default_value=None),
                Parameter("extra", default_value=None),
            ],
            name="assert_is",
            implementation=_assert_is_impl,
        ),
        qcore.asserts.assert_is_not: ExtendedArgSpec(
            [
                Parameter("expected"),
                Parameter("actual"),
                Parameter("message", default_value=None),
                Parameter("extra", default_value=None),
            ],
            name="assert_is_not",
            implementation=_assert_is_not_impl,
        ),
        # Need to override this because the type for the tp parameter in typeshed is too strict
        NewType: ExtendedArgSpec(
            [Parameter("name", typ=TypedValue(str)), Parameter(name="tp")],
            name="NewType",
        ),
        type.__subclasses__: ExtendedArgSpec(
            [Parameter("self")],
            name="type.__subclasses__",
            implementation=_subclasses_impl,
        ),
    }

    def __init__(self, config: Config) -> None:
        self.config = config
        self.ts_finder = TypeshedFinder(verbose=False)
        self.known_argspecs = {}
        self.generic_bases_cache = {}
        default_argspecs = dict(self.DEFAULT_ARGSPECS)
        default_argspecs.update(self.config.get_known_argspecs(self))

        for obj, argspec in default_argspecs.items():
            self.known_argspecs[obj] = argspec

    def __reduce_ex__(self, proto: object) -> object:
        # Don't pickle the actual argspecs, which are frequently unpicklable.
        return self.__class__, (self.config,)

    def from_argspec(
        self,
        argspec: inspect.Signature,
        *,
        kwonly_args: Any = None,
        name: Optional[str] = None,
        logger: Optional[Logger] = None,
        implementation: Optional[ImplementationFn] = None,
        function_object: Optional[object] = None,
    ) -> ExtendedArgSpec:
        """Constructs an ExtendedArgSpec from a standard argspec.

        argspec is an inspect.Signature.

        kwonly_args may be a list of custom keyword-only arguments added to the argspec or None.

        name is the name of the function. This is used for better error messages.

        logger is the log function to be used.

        implementation is an implementation function for this object.

        TODO: do we need support for non-Signature argspecs and for the separate kwonly_args
        argument?

        """
        if kwonly_args is None:
            kwonly_args = []
        else:
            kwonly_args = list(kwonly_args)
        func_globals = getattr(function_object, "__globals__", None)

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
        for i, parameter in enumerate(argspec.parameters.values()):
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                starargs = parameter.name
            elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                kwargs = parameter.name
            elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                kwonly_args.append(
                    self._parameter_from_signature(
                        parameter, func_globals, function_object, i
                    )
                )
            else:
                # positional or positional-or-keyword
                args.append(
                    self._parameter_from_signature(
                        parameter, func_globals, function_object, i
                    )
                )
        return ExtendedArgSpec(
            args,
            starargs=starargs,
            kwargs=kwargs,
            kwonly_args=kwonly_args,
            return_value=return_value,
            name=name,
            logger=logger,
            implementation=implementation,
        )

    def _parameter_from_signature(
        self,
        parameter: inspect.Parameter,
        func_globals: Mapping[str, object],
        function_object: Optional[object],
        index: int,
    ) -> Parameter:
        """Given an inspect.Parameter, returns a Parameter object."""
        typ = self._get_type_for_parameter(
            parameter, func_globals, function_object, index
        )
        if parameter.default is inspect.Parameter.empty:
            default_value = Parameter.no_default_value
        elif parameter.default is Parameter.no_default_value:
            # hack to prevent errors in code that use Parameter
            default_value = object()
        else:
            default_value = parameter.default
        return Parameter(parameter.name, default_value=default_value, typ=typ)

    def _get_type_for_parameter(
        self,
        parameter: inspect.Parameter,
        func_globals: Mapping[str, object],
        function_object: Optional[object],
        index: int,
    ) -> Optional[Value]:
        if parameter.annotation is not inspect.Parameter.empty:
            return type_from_runtime(parameter.annotation, globals=func_globals)
        # If this is the self argument of a method, try to infer the self type.
        elif index == 0 and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            module_name = getattr(function_object, "__module__", None)
            qualname = getattr(function_object, "__qualname__", None)
            name = getattr(function_object, "__name__", None)
            if (
                qualname != name
                and module_name is not None
                and module_name in sys.modules
            ):
                module = sys.modules[module_name]
                *class_names, function_name = qualname.split(".")
                class_obj = module
                for class_name in class_names:
                    class_obj = getattr(class_obj, class_name, None)
                    if class_obj is None:
                        break
                if (
                    class_obj is not None
                    and inspect.getattr_static(class_obj, function_name, None)
                    is function_object
                ):
                    return type_from_maybe_generic(class_obj)
        return VariableNameValue.from_varname(
            parameter.name, self.config.varname_value_map()
        )

    def get_argspec(
        self,
        obj: object,
        name: Optional[str] = None,
        logger: Optional[Logger] = None,
        implementation: Optional[ImplementationFn] = None,
    ) -> MaybeArgspec:
        """Constructs the ExtendedArgSpec for a Python object."""
        kwargs = {"name": name, "logger": logger, "implementation": implementation}
        argspec = self._cached_get_argspec(obj, kwargs)
        return argspec

    def _cached_get_argspec(
        self, obj: object, kwargs: Mapping[str, Any]
    ) -> MaybeArgspec:
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

    def _uncached_get_argspec(
        self, obj: Any, kwargs: Mapping[str, Any]
    ) -> MaybeArgspec:
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
            if (
                asyncio.iscoroutinefunction(obj)
                and argspec.return_value is UNRESOLVED_VALUE
            ):
                argspec.return_value = GenericValue(Awaitable, [UNRESOLVED_VALUE])
            return argspec

        if inspect.isfunction(obj):
            if hasattr(obj, "inner"):
                # @qclient.task_queue.exec_after_request() puts the original function in .inner
                return self._cached_get_argspec(obj.inner, kwargs)

            argspec = self.from_argspec(
                self._safe_get_argspec(obj), function_object=obj, **kwargs
            )
            if asyncio.iscoroutinefunction(obj):
                argspec.return_value = GenericValue(Awaitable, [argspec.return_value])
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
            if argspec is None:
                return None

            kwonly_args = []
            for cls_, args in self.config.CLASS_TO_KEYWORD_ONLY_ARGUMENTS.items():
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

    def _safe_get_argspec(self, obj: Any) -> Optional[inspect.Signature]:
        """Wrapper around inspect.getargspec that catches TypeErrors."""
        try:
            # follow_wrapped=True leads to problems with decorators that
            # mess with the arguments, such as mock.patch.
            sig = inspect.signature(obj, follow_wrapped=False)
        except (TypeError, ValueError, AttributeError):
            # TypeError if signature() does not support the object, ValueError
            # if it cannot provide a signature, and AttributeError if we're on
            # Python 2.
            return None
        else:
            # Signature preserves the return annotation for wrapped functions,
            # because @functools.wraps copies the __annotations__ of the wrapped function. We
            # don't want that, because the wrapper may have changed the return type.
            # This caused problems with @contextlib.contextmanager.
            if _safe_has_attr(obj, "__wrapped__"):
                return sig.replace(return_annotation=inspect.Signature.empty)
            else:
                return sig

    def get_generic_bases(
        self, typ: type, generic_args: Sequence[Value] = ()
    ) -> Dict[type, Sequence[Value]]:
        if typ is Generic or is_typing_name(typ, "Protocol"):
            return {}
        generic_bases = self._get_generic_bases_cached(typ)
        if typ not in generic_bases:
            return generic_bases
        my_typevars = generic_bases[typ]
        if not my_typevars:
            return generic_bases
        tv_map = {}
        for i, tv_value in enumerate(my_typevars):
            if not isinstance(tv_value, TypeVarValue):
                continue
            try:
                value = generic_args[i]
            except IndexError:
                value = UNRESOLVED_VALUE
            tv_map[tv_value.typevar] = value
        return {
            base: substitute_typevars(args, tv_map)
            for base, args in generic_bases.items()
        }

    def _get_generic_bases_cached(self, typ: type) -> Dict[type, Sequence[Value]]:
        try:
            return self.generic_bases_cache[typ]
        except KeyError:
            pass
        except Exception:
            return {}  # We don't support unhashable types.
        bases = self.ts_finder.get_bases(typ)
        generic_bases = self._extract_bases(typ, bases)
        if generic_bases is None:
            bases = [type_from_runtime(base) for base in self.get_runtime_bases(typ)]
            generic_bases = self._extract_bases(typ, bases)
            assert (
                generic_bases is not None
            ), f"failed to extract runtime bases from {typ}"
        return generic_bases

    def _extract_bases(
        self, typ: type, bases: Optional[Sequence[Value]]
    ) -> Optional[Dict[type, Sequence[Value]]]:
        if bases is None:
            return None
        my_typevars = uniq_chain(extract_typevars(base) for base in bases)
        generic_bases = {}
        generic_bases[typ] = [TypeVarValue(tv) for tv in my_typevars]
        for base in bases:
            if isinstance(base, TypedValue):
                assert base.typ is not typ, base
                if isinstance(base, GenericValue):
                    args = base.args
                else:
                    args = ()
                generic_bases.update(self.get_generic_bases(base.typ, args))
            else:
                return None
        return generic_bases

    def get_runtime_bases(self, typ: type) -> Sequence[Value]:
        if typing_inspect.is_generic_type(typ):
            return typing_inspect.get_generic_bases(typ)
        return typ.__bases__


@dataclass
class _AnnotationContext(Context):
    finder: "TypeshedFinder"
    module: str

    def show_error(
        self, message: str, error_code: ErrorCode = ErrorCode.invalid_annotation
    ) -> None:
        self.finder.log(message, ())

    def get_name(self, node: ast.AST) -> Value:
        info = self.finder._get_info_for_name("%s.%s" % (self.module, node.id))
        if info is not None:
            return self.finder._value_from_info(info, self.module)
        elif hasattr(builtins, node.id):
            val = getattr(builtins, node.id)
            if val is None or isinstance(val, type):
                return KnownValue(val)
        return UNRESOLVED_VALUE


# These are specified as just "List = _Alias()" in typing.pyi. Redirect
# them to the proper runtime equivalent.
_TYPING_ALIASES = {
    "typing.List": "builtins.list",
    "typing.Dict": "builtins.dict",
    "typing.DefaultDict": "collections.defaultdict",
    "typing.Set": "builtins.set",
    "typing.Frozenzet": "builtins.frozenset",
    "typing.Counter": "collections.Counter",
    "typing.Deque": "collections.deque",
    "typing.ChainMap": "collections.ChainMap",
    "typing.OrderedDict": "collections.OrderedDict",
    "typing.Tuple": "builtins.tuple",
}


class TypeshedFinder(object):
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.resolver = typeshed_client.Resolver(version=sys.version_info[:2])
        self._assignment_cache = {}

    def log(self, message: str, obj: object) -> None:
        if not self.verbose:
            return
        print("%s: %r" % (message, obj))

    def get_argspec(self, obj: object) -> Optional[ExtendedArgSpec]:
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
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        argspec = self._get_argspec_from_info(info, obj, fq_name, mod)
        if argspec is not None:
            self.log("Found argspec", (fq_name, argspec))
        return argspec

    def get_bases(self, typ: type) -> Optional[List[Value]]:
        # The way AbstractSet/Set is handled between collections and typing is
        # too confusing, just hardcode it.
        if typ is AbstractSet:
            return [GenericValue(Collection, (TypeVarValue(T_co),))]
        if typ is AbstractContextManager:
            return [GenericValue(Generic, (TypeVarValue(T_co),))]
        if typ is Callable or typ is collections.abc.Callable:
            return None
        fq_name = self._get_fq_name(typ)
        if fq_name is None:
            return None
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        return self._get_bases_from_info(info, mod)

    def _get_bases_from_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str
    ) -> Optional[List[Value]]:
        if info is None:
            return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_bases_from_info(info.info, ".".join(info.source_module))
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast3.ClassDef):
                bases = info.ast.bases
                return [self._parse_expr(base, mod) for base in bases]
            elif isinstance(info.ast, ast3.Assign):
                val = self._parse_expr(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    return self.get_bases(val.val)
                else:
                    return [val]
            elif isinstance(
                info.ast, (typeshed_client.OverloadedName, typeshed_client.ImportedName)
            ):
                return None  # overloads are not supported yet
            else:
                raise NotImplementedError(ast3.dump(info.ast))
        return None

    def _get_method_argspec_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        obj: object,
        fq_name: str,
        mod: str,
    ) -> Optional[ExtendedArgSpec]:
        if info is None:
            return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_method_argspec_from_info(
                info.info, obj, fq_name, ".".join(info.source_module)
            )
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

    def _get_fq_name(self, obj: Any) -> Optional[str]:
        if obj is GeneratorType:
            return "typing.Generator"
        if IS_PRE_38:
            if obj is Sized:
                return "typing.Sized"
        try:
            module = obj.__module__
            if module is None:
                module = "builtins"
            fq_name = ".".join([module, obj.__qualname__])
            return _TYPING_ALIASES.get(fq_name, fq_name)
        except (AttributeError, TypeError):
            self.log("Ignoring object without module or qualname", obj)
            return None

    def _get_argspec_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        obj: object,
        fq_name: str,
        mod: str,
    ) -> Optional[ExtendedArgSpec]:
        if isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast3.FunctionDef):
                return self._get_argspec_from_func_def(
                    info.ast, obj, mod, is_async_fn=False
                )
            elif isinstance(info.ast, ast3.AsyncFunctionDef):
                return self._get_argspec_from_func_def(
                    info.ast, obj, mod, is_async_fn=True
                )
            else:
                self.log("Ignoring unrecognized AST", (fq_name, info))
                return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_argspec_from_info(
                info.info, obj, fq_name, ".".join(info.source_module)
            )
        elif info is None:
            return None
        else:
            self.log("Ignoring unrecognized info", (fq_name, info))
            return None

    @qcore.caching.cached_per_instance()
    def _get_info_for_name(self, fq_name: str) -> typeshed_client.resolver.ResolvedName:
        return self.resolver.get_fully_qualified_name(fq_name)

    def _get_argspec_from_func_def(
        self,
        node: Union[ast3.FunctionDef, ast3.AsyncFunctionDef],
        obj: object,
        mod: str,
        is_async_fn: bool,
    ) -> Optional[ExtendedArgSpec]:
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
        kwonly = list(self._parse_param_list(args.kwonlyargs, args.kw_defaults, mod))
        return ExtendedArgSpec(
            arguments,
            starargs=starargs,
            kwargs=kwargs,
            kwonly_args=kwonly,
            return_value=GenericValue(Awaitable, [return_value])
            if is_async_fn
            else return_value,
            name=obj.__name__,
        )

    def _parse_param_list(
        self,
        args: Iterable[ast3.arg],
        defaults: Iterable[Optional[ast3.AST]],
        module: str,
    ) -> Iterable[Parameter]:
        for arg, default in zip(args, defaults):
            typ = None
            if arg.annotation is not None:
                typ = self._parse_expr(arg.annotation, module)

            if default is None:
                yield Parameter(arg.arg, typ=typ)
            else:
                # doesn't matter what the default is
                yield Parameter(arg.arg, typ=typ, default_value=None)

    def _parse_expr(self, node: ast3.AST, module: str) -> Value:
        ctx = _AnnotationContext(finder=self, module=module)
        typ = type_from_ast(cast(ast.AST, node), ctx=ctx)
        if self.verbose and typ is UNRESOLVED_VALUE:
            self.log("Got UNRESOLVED_VALUE", (ast3.dump(node), module))
        return typ

    def _value_from_info(
        self, info: typeshed_client.resolver.ResolvedName, module: str
    ) -> Value:
        if isinstance(info, typeshed_client.ImportedInfo):
            return self._value_from_info(info.info, ".".join(info.source_module))
        elif isinstance(info, typeshed_client.NameInfo):
            fq_name = f"{module}.{info.name}"
            if fq_name in _TYPING_ALIASES:
                new_fq_name = _TYPING_ALIASES[fq_name]
                info = self._get_info_for_name(new_fq_name)
                return self._value_from_info(
                    info, new_fq_name.rsplit(".", maxsplit=1)[0]
                )
            elif IS_PRE_38:
                if fq_name in ("typing.Protocol", "typing_extensions.Protocol"):
                    return KnownValue(Protocol)
            if isinstance(info.ast, ast3.Assign):
                key = (module, info.ast)
                if key in self._assignment_cache:
                    return self._assignment_cache[key]
                value = self._parse_expr(info.ast.value, module)
                self._assignment_cache[key] = value
                return value
            try:
                __import__(module)
                mod = sys.modules[module]
                return KnownValue(getattr(mod, info.name))
            except Exception:
                self.log("Unable to import", (module, info))
                return UNRESOLVED_VALUE
        elif isinstance(info, tuple):
            module_path = ".".join(info)
            try:
                __import__(module_path)
                return KnownValue(sys.modules[module_path])
            except Exception:
                self.log("Unable to import", module_path)
                return UNRESOLVED_VALUE
        else:
            self.log("Ignoring info", info)
            return UNRESOLVED_VALUE


def _is_qcore_decorator(obj: object) -> bool:
    try:
        return (
            hasattr(obj, "is_decorator")
            and obj.is_decorator()
            and hasattr(obj, "decorator")
        )
    except Exception:
        # black.Line has an is_decorator attribute but it is not a method
        return False
