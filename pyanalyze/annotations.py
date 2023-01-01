"""

Code for understanding type annotations.

This file contains functions that turn various representations of
Python type annotations into :class:`pyanalyze.value.Value` objects.

There are three major functions:

- :func:`type_from_runtime` takes a runtime Python object, for example
  ``type_from_value(int)`` -> ``TypedValue(int)``.
- :func:`type_from_value` takes an existing :class:`pyanalyze.value.Value`
  object. For example, evaluating the expression ``int`` will produce
  ``KnownValue(int)``, and calling :func:`type_from_value` on that value
  will produce ``TypedValue(int)``.
- :func:`type_from_ast` takes an AST node and evaluates it into a type.

These functions all rely on each other. For example, when a forward
reference is found in a runtime annotation, the code parses it and calls
:func:`type_from_ast` to evaluate it.

These functions all use :class:`Context` objects to resolve names and
show errors.

"""
import ast
import builtins
import contextlib
import typing
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field, InitVar
from typing import (
    Any,
    cast,
    Container,
    ContextManager,
    Generator,
    Mapping,
    NamedTuple,
    NewType,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import qcore

import typing_inspect
from typing_extensions import ParamSpec, TypedDict

from . import type_evaluation

from .error_code import ErrorCode
from .extensions import (
    AsynqCallable,
    CustomCheck,
    ExternalType,
    HasAttrGuard,
    NoReturnGuard,
    ParameterTypeGuard,
    TypeGuard,
    deprecated,
)
from .find_unused import used
from .functions import FunctionDefNode
from .node_visitor import ErrorContext
from .safe import is_instance_of_typing_name, is_typing_name
from .signature import (
    ELLIPSIS_PARAM,
    InvalidSignature,
    ParameterKind,
    Signature,
    SigParameter,
)
from .value import (
    _HashableValue,
    annotate_value,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CustomCheckExtension,
    Extension,
    GenericValue,
    HasAttrGuardExtension,
    KnownValue,
    MultiValuedValue,
    NewTypeValue,
    NO_RETURN_VALUE,
    NoReturnGuardExtension,
    ParameterTypeGuardExtension,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    SelfTVV,
    SequenceValue,
    SubclassValue,
    TypedDictValue,
    TypedValue,
    TypeGuardExtension,
    TypeVarLike,
    TypeVarValue,
    unite_values,
    UnpackedValue,
    Value,
)

if TYPE_CHECKING:
    from .name_check_visitor import NameCheckVisitor

try:
    from types import GenericAlias
    from typing import get_args, get_origin  # Python 3.9
except ImportError:
    GenericAlias = None

    def get_origin(obj: object) -> Any:
        return None

    def get_args(obj: object) -> Tuple[Any, ...]:
        return ()


CONTEXT_MANAGER_TYPES = (typing.ContextManager, contextlib.AbstractContextManager)
ASYNC_CONTEXT_MANAGER_TYPES = (
    typing.AsyncContextManager,
    contextlib.AbstractAsyncContextManager,
)


@dataclass
class Context:
    """A context for evaluating annotations.

    The base implementation does very little. Subclass this to do something more useful.

    """

    should_suppress_undefined_names: bool = field(default=False, init=False)
    """While this is True, no errors are shown for undefined names."""
    _being_evaluated: Set[int] = field(default_factory=set, init=False)

    def suppress_undefined_names(self) -> ContextManager[None]:
        """Temporarily suppress errors about undefined names."""
        return qcore.override(self, "should_suppress_undefined_names", True)

    def is_being_evaluted(self, obj: object) -> bool:
        return id(obj) in self._being_evaluated

    @contextlib.contextmanager
    def add_evaluation(self, obj: object) -> Generator[None, None, None]:
        """Temporarily add an object to the set of objects being evaluated.

        This is used to prevent infinite recursion when evaluating forward references.

        """
        obj_id = id(obj)
        self._being_evaluated.add(obj_id)
        try:
            yield
        finally:
            self._being_evaluated.remove(obj_id)

    def show_error(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.invalid_annotation,
        node: Optional[ast.AST] = None,
    ) -> None:
        """Show an error found while evaluating an annotation."""
        pass

    def get_name(self, node: ast.Name) -> Value:
        """Return the :class:`Value <pyanalyze.value.Value>` corresponding to a name."""
        return AnyValue(AnySource.inference)

    def handle_undefined_name(self, name: str) -> Value:
        if self.should_suppress_undefined_names:
            return AnyValue(AnySource.inference)
        self.show_error(
            f"Undefined name {name!r} used in annotation", ErrorCode.undefined_name
        )
        return AnyValue(AnySource.error)

    def get_name_from_globals(self, name: str, globals: Mapping[str, Any]) -> Value:
        if name in globals:
            return KnownValue(globals[name])
        elif hasattr(builtins, name):
            return KnownValue(getattr(builtins, name))
        return self.handle_undefined_name(name)

    def get_attribute(self, root_value: Value, node: ast.Attribute) -> Value:
        if isinstance(root_value, KnownValue):
            try:
                return KnownValue(getattr(root_value.val, node.attr))
            except AttributeError:
                self.show_error(
                    f"{root_value.val!r} has no attribute {node.attr!r}", node=node
                )
                return AnyValue(AnySource.error)
        elif not isinstance(root_value, AnyValue):
            self.show_error(f"Cannot resolve annotation {root_value}", node=node)
        return AnyValue(AnySource.error)


@dataclass
class RuntimeEvaluator(type_evaluation.Evaluator, Context):
    globals: Mapping[str, object] = field(repr=False)
    func: typing.Callable[..., Any]

    def evaluate_type(self, node: ast.AST) -> Value:
        return type_from_ast(node, ctx=self)

    def evaluate_value(self, node: ast.AST) -> Value:
        return value_from_ast(node, ctx=self, error_on_unrecognized=False)

    def get_name(self, node: ast.Name) -> Value:
        """Return the :class:`Value <pyanalyze.value.Value>` corresponding to a name."""
        return self.get_name_from_globals(node.id, self.globals)


@dataclass
class SyntheticEvaluator(type_evaluation.Evaluator):
    error_ctx: ErrorContext
    annotations_context: Context

    def show_error(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.invalid_annotation,
        node: Optional[ast.AST] = None,
    ) -> None:
        self.error_ctx.show_error(node or self.node, message, error_code=error_code)

    def evaluate_type(self, node: ast.AST) -> Value:
        return type_from_ast(node, ctx=self.annotations_context)

    def evaluate_value(self, node: ast.AST) -> Value:
        return value_from_ast(
            node, ctx=self.annotations_context, error_on_unrecognized=False
        )

    def get_name(self, node: ast.Name) -> Value:
        """Return the :class:`Value <pyanalyze.value.Value>` corresponding to a name."""
        return self.annotations_context.get_name(node)

    @classmethod
    def from_visitor(
        cls,
        node: FunctionDefNode,
        visitor: "NameCheckVisitor",
        return_annotation: Value,
    ) -> "SyntheticEvaluator":
        return cls(
            node,
            return_annotation,
            visitor,
            _DefaultContext(visitor, node, use_name_node_for_error=True),
        )


@used  # part of an API
def type_from_ast(
    ast_node: ast.AST,
    visitor: Optional["NameCheckVisitor"] = None,
    ctx: Optional[Context] = None,
) -> Value:
    """Given an AST node representing an annotation, return a
    :class:`Value <pyanalyze.value.Value>`.

    :param ast_node: AST node to evaluate.

    :param visitor: Visitor class to use. This is used in the default
                    :class:`Context` to resolve names and show errors.
                    This is ignored if `ctx` is given.

    :param ctx: :class:`Context` to use for evaluation.

    """
    if ctx is None:
        ctx = _DefaultContext(visitor, ast_node)
    return _type_from_ast(ast_node, ctx)


def type_from_annotations(
    annotations: Mapping[str, object],
    key: str,
    *,
    globals: Optional[Mapping[str, object]] = None,
    ctx: Optional[Context] = None,
) -> Optional[Value]:
    try:
        annotation = annotations[key]
    except Exception:
        # Malformed __annotations__
        return None
    else:
        maybe_val = type_from_runtime(annotation, globals=globals, ctx=ctx)
        if maybe_val != AnyValue(AnySource.incomplete_annotation):
            return maybe_val
    return None


def type_from_runtime(
    val: object,
    visitor: Optional["NameCheckVisitor"] = None,
    node: Optional[ast.AST] = None,
    globals: Optional[Mapping[str, object]] = None,
    ctx: Optional[Context] = None,
    *,
    allow_unpack: bool = False,
) -> Value:
    """Given a runtime annotation object, return a
    :class:`Value <pyanalyze.value.Value>`.

    :param val: Object to evaluate. This will usually come from an
                ``__annotations__`` dictionary.

    :param visitor: Visitor class to use. This is used in the default
                    :class:`Context` to resolve names and show errors.
                    This is ignored if `ctx` is given.

    :param node: AST node that the annotation derives from. This is
                 used for showing errors. Ignored if `ctx` is given.

    :param globals: Dictionary of global variables that can be used
                    to resolve names. Ignored if `ctx` is given.

    :param ctx: :class:`Context` to use for evaluation.

    :param allow_unpack: Whether to allow `Unpack` types.

    """

    if ctx is None:
        ctx = _DefaultContext(visitor, node, globals)
    return _type_from_runtime(val, ctx, allow_unpack=allow_unpack)


def type_from_value(
    value: Value,
    visitor: Optional["NameCheckVisitor"] = None,
    node: Optional[ast.AST] = None,
    ctx: Optional[Context] = None,
    *,
    is_typeddict: bool = False,
    allow_unpack: bool = False,
) -> Value:
    """Given a :class:`Value <pyanalyze.value.Value` representing an annotation,
    return a :class:`Value <pyanalyze.value.Value>` representing the type.

    The input value represents an expression, the output value represents
    a type. For example, the :term:`impl` of ``typing.cast(typ, val)``
    calls :func:`type_from_value` on the value it receives for its
    `typ` argument and returns the result.

    :param value: :class:`Value <pyanalyze.value.Value` to evaluate.

    :param visitor: Visitor class to use. This is used in the default
                    :class:`Context` to resolve names and show errors.
                    This is ignored if `ctx` is given.

    :param node: AST node that the annotation derives from. This is
                 used for showing errors. Ignored if `ctx` is given.

    :param ctx: :class:`Context` to use for evaluation.

    :param is_typeddict: Whether we are at the top level of a `TypedDict`
                         definition.

    """
    if ctx is None:
        ctx = _DefaultContext(visitor, node)
    return _type_from_value(
        value, ctx, is_typeddict=is_typeddict, allow_unpack=allow_unpack
    )


def value_from_ast(
    ast_node: ast.AST, ctx: Context, *, error_on_unrecognized: bool = True
) -> Value:
    val = _Visitor(ctx).visit(ast_node)
    if val is None:
        if error_on_unrecognized:
            ctx.show_error("Invalid type annotation", node=ast_node)
        return AnyValue(AnySource.error)
    return val


def _type_from_ast(
    node: ast.AST,
    ctx: Context,
    *,
    is_typeddict: bool = False,
    allow_unpack: bool = False,
) -> Value:
    val = value_from_ast(node, ctx)
    return _type_from_value(
        val, ctx, is_typeddict=is_typeddict, allow_unpack=allow_unpack
    )


def _type_from_runtime(
    val: Any, ctx: Context, *, is_typeddict: bool = False, allow_unpack: bool = False
) -> Value:
    if isinstance(val, str):
        return _eval_forward_ref(
            val, ctx, is_typeddict=is_typeddict, allow_unpack=allow_unpack
        )
    elif GenericAlias is not None and isinstance(val, GenericAlias):
        origin = get_origin(val)
        args = get_args(val)
        if origin is tuple and not args:
            return SequenceValue(tuple, [])
        return _value_of_origin_args(
            origin, args, val, ctx, allow_unpack=origin is tuple
        )
    elif typing_inspect.is_literal_type(val):
        args = typing_inspect.get_args(val)
        if len(args) == 0:
            return KnownValue(args[0])
        else:
            return unite_values(*[KnownValue(arg) for arg in args])
    elif typing_inspect.is_union_type(val):
        args = typing_inspect.get_args(val)
        return unite_values(*[_type_from_runtime(arg, ctx) for arg in args])
    elif typing_inspect.is_tuple_type(val):
        args = typing_inspect.get_args(val)
        if not args:
            if val is tuple or val is Tuple:
                return TypedValue(tuple)
            else:
                return SequenceValue(tuple, [])
        elif len(args) == 2 and args[1] is Ellipsis:
            return GenericValue(tuple, [_type_from_runtime(args[0], ctx)])
        elif len(args) == 1 and args[0] == ():
            return SequenceValue(tuple, [])  # empty tuple
        else:
            return _make_sequence_value(
                tuple,
                [_type_from_runtime(arg, ctx, allow_unpack=True) for arg in args],
                ctx,
            )
    elif is_instance_of_typing_name(val, "_TypedDictMeta"):
        required_keys = getattr(val, "__required_keys__", None)
        # 3.8's typing.TypedDict doesn't have __required_keys__. With
        # inheritance, this makes it apparently impossible to figure out which
        # keys are required at runtime.
        total = getattr(val, "__total__", True)
        if hasattr(val, "__extra_keys__"):
            extra_keys = _type_from_runtime(val.__extra_keys__, ctx)
        else:
            extra_keys = None
        return TypedDictValue(
            {
                key: _get_typeddict_value(value, ctx, key, required_keys, total)
                for key, value in val.__annotations__.items()
            },
            extra_keys=extra_keys,
        )
    elif val is InitVar:
        # On 3.6 and 3.7, InitVar[T] just returns InitVar at runtime, so we can't
        # get the actual type out.
        return AnyValue(AnySource.inference)
    elif isinstance(val, InitVar):
        # val.type exists only on 3.8+, but on earlier versions
        # InitVar instances aren't being created
        # static analysis: ignore[undefined_attribute]
        return type_from_runtime(val.type)
    elif is_instance_of_typing_name(val, "_AnnotatedAlias"):
        # Annotated in typing and newer typing_extensions
        return _make_annotated(
            _type_from_runtime(val.__origin__, ctx),
            [KnownValue(v) for v in val.__metadata__],
            ctx,
        )
    elif typing_inspect.is_generic_type(val):
        origin = typing_inspect.get_origin(val)
        args = typing_inspect.get_args(val)
        if getattr(val, "_special", False):
            args = []  # distinguish List from List[T] on 3.7 and 3.8
        return _value_of_origin_args(
            origin,
            args,
            val,
            ctx,
            is_typeddict=is_typeddict,
            allow_unpack=allow_unpack or origin is tuple or origin is Tuple,
        )
    elif typing_inspect.is_callable_type(val):
        args = typing_inspect.get_args(val)
        return _value_of_origin_args(Callable, args, val, ctx)
    elif val is AsynqCallable:
        return CallableValue(Signature.make([ELLIPSIS_PARAM], is_asynq=True))
    elif val is typing.Any:
        return AnyValue(AnySource.explicit)
    elif isinstance(val, type):
        return _maybe_typed_value(val)
    elif val is None:
        return KnownValue(None)
    elif is_typing_name(val, "NoReturn") or is_typing_name(val, "Never"):
        return NO_RETURN_VALUE
    elif is_typing_name(val, "Self"):
        return SelfTVV
    elif is_typing_name(val, "LiteralString"):
        return TypedValue(str, literal_only=True)
    elif hasattr(val, "__supertype__"):
        if isinstance(val.__supertype__, type):
            # NewType
            return NewTypeValue(val)
        elif typing_inspect.is_tuple_type(val.__supertype__):
            # TODO figure out how to make NewTypes over tuples work
            return AnyValue(AnySource.inference)
        else:
            ctx.show_error(f"Invalid NewType {val}")
            return AnyValue(AnySource.error)
    elif typing_inspect.is_typevar(val):
        tv = cast(TypeVar, val)
        return make_type_var_value(tv, ctx)
    elif is_instance_of_typing_name(val, "ParamSpec"):
        return TypeVarValue(val, is_paramspec=True)
    elif is_instance_of_typing_name(val, "ParamSpecArgs"):
        return ParamSpecArgsValue(val.__origin__)
    elif is_instance_of_typing_name(val, "ParamSpecKwargs"):
        return ParamSpecKwargsValue(val.__origin__)
    elif is_typing_name(val, "Final") or is_typing_name(val, "ClassVar"):
        return AnyValue(AnySource.incomplete_annotation)
    elif typing_inspect.is_classvar(val) or typing_inspect.is_final_type(val):
        typ = val.__args__[0]
        return _type_from_runtime(typ, ctx)
    elif is_instance_of_typing_name(val, "_ForwardRef") or is_instance_of_typing_name(
        val, "ForwardRef"
    ):
        if ctx.is_being_evaluted(val):
            return AnyValue(AnySource.inference)
        with ctx.add_evaluation(val):
            # This is necessary because the forward ref may be defined in a different file, in
            # which case we don't know which names are valid in it.
            with ctx.suppress_undefined_names():
                return _eval_forward_ref(
                    val.__forward_arg__, ctx, is_typeddict=is_typeddict
                )
    elif val is Ellipsis:
        # valid in Callable[..., ]
        return AnyValue(AnySource.explicit)
    elif is_instance_of_typing_name(val, "_TypeAlias"):
        # typing.Pattern and Match, which are not normal generic types for some reason
        return GenericValue(val.impl_type, [_type_from_runtime(val.type_var, ctx)])
    elif isinstance(val, TypeGuard):
        return AnnotatedValue(
            TypedValue(bool),
            [TypeGuardExtension(_type_from_runtime(val.guarded_type, ctx))],
        )
    elif isinstance(val, AsynqCallable):
        params = _callable_args_from_runtime(val.args, "AsynqCallable", ctx)
        sig = Signature.make(
            params, _type_from_runtime(val.return_type, ctx), is_asynq=True
        )
        return CallableValue(sig)
    elif isinstance(val, ExternalType):
        try:
            typ = qcore.helpers.object_from_string(val.type_path)
        except Exception:
            ctx.show_error(f"Cannot resolve type {val.type_path!r}")
            return AnyValue(AnySource.error)
        return _type_from_runtime(typ, ctx)
    elif is_typing_name(val, "TypeAlias"):
        return AnyValue(AnySource.incomplete_annotation)
    elif is_typing_name(val, "TypedDict"):
        return KnownValue(TypedDict)
    else:
        origin = get_origin(val)
        if isinstance(origin, type):
            return _maybe_typed_value(origin)
        elif val is NamedTuple:
            return TypedValue(tuple)
        ctx.show_error(f"Invalid type annotation {val}")
        return AnyValue(AnySource.error)


def make_type_var_value(tv: TypeVarLike, ctx: Context) -> TypeVarValue:
    if tv.__bound__ is not None:
        bound = _type_from_runtime(tv.__bound__, ctx)
    else:
        bound = None
    if isinstance(tv, TypeVar) and tv.__constraints__:
        constraints = tuple(
            _type_from_runtime(constraint, ctx) for constraint in tv.__constraints__
        )
    else:
        constraints = ()
    return TypeVarValue(tv, bound=bound, constraints=constraints)


def _callable_args_from_runtime(
    arg_types: Any, label: str, ctx: Context
) -> Sequence[SigParameter]:
    if arg_types is Ellipsis or arg_types == [Ellipsis]:
        return [ELLIPSIS_PARAM]
    elif type(arg_types) in (tuple, list):
        if len(arg_types) == 1:
            (arg,) = arg_types
            if arg is Ellipsis:
                return [ELLIPSIS_PARAM]
            elif is_typing_name(getattr(arg, "__origin__", None), "Concatenate"):
                return _args_from_concatenate(arg, ctx)
            elif is_instance_of_typing_name(arg, "ParamSpec"):
                param_spec = TypeVarValue(arg, is_paramspec=True)
                param = SigParameter(
                    "__P", kind=ParameterKind.PARAM_SPEC, annotation=param_spec
                )
                return [param]
        types = [_type_from_runtime(arg, ctx) for arg in arg_types]
        params = [
            SigParameter(
                f"@{i}",
                kind=ParameterKind.PARAM_SPEC
                if isinstance(typ, TypeVarValue) and typ.is_paramspec
                else ParameterKind.POSITIONAL_ONLY,
                annotation=typ,
            )
            for i, typ in enumerate(types)
        ]
        return params
    elif is_instance_of_typing_name(arg_types, "ParamSpec"):
        param_spec = TypeVarValue(arg_types, is_paramspec=True)
        param = SigParameter(
            "__P", kind=ParameterKind.PARAM_SPEC, annotation=param_spec
        )
        return [param]
    elif is_typing_name(getattr(arg_types, "__origin__", None), "Concatenate"):
        return _args_from_concatenate(arg_types, ctx)
    else:
        ctx.show_error(f"Invalid arguments to {label}: {arg_types!r}")
        return [ELLIPSIS_PARAM]


def _args_from_concatenate(concatenate: Any, ctx: Context) -> Sequence[SigParameter]:
    types = [_type_from_runtime(arg, ctx) for arg in concatenate.__args__]
    params = [
        SigParameter(
            f"@{i}",
            kind=ParameterKind.PARAM_SPEC
            if i == len(types) - 1
            else ParameterKind.POSITIONAL_ONLY,
            annotation=annotation,
        )
        for i, annotation in enumerate(types)
    ]
    return params


def _get_typeddict_value(
    value: Value,
    ctx: Context,
    key: str,
    required_keys: Optional[Container[str]],
    total: bool,
) -> Tuple[bool, Value]:
    val = _type_from_runtime(value, ctx, is_typeddict=True)
    if isinstance(val, Pep655Value):
        return (val.required, val.value)
    if required_keys is None:
        required = total
    else:
        required = key in required_keys
    return required, val


def _eval_forward_ref(
    val: str, ctx: Context, *, is_typeddict: bool = False, allow_unpack: bool = False
) -> Value:
    try:
        tree = ast.parse(val, mode="eval")
    except SyntaxError:
        ctx.show_error(f"Syntax error in type annotation: {val}")
        return AnyValue(AnySource.error)
    else:
        return _type_from_ast(
            tree.body, ctx, is_typeddict=is_typeddict, allow_unpack=allow_unpack
        )


def _type_from_value(
    value: Value,
    ctx: Context,
    *,
    is_typeddict: bool = False,
    allow_unpack: bool = False,
) -> Value:
    if isinstance(value, KnownValue):
        return _type_from_runtime(
            value.val, ctx, is_typeddict=is_typeddict, allow_unpack=allow_unpack
        )
    elif isinstance(value, TypeVarValue):
        return value
    elif isinstance(value, MultiValuedValue):
        return unite_values(
            *[
                _type_from_value(
                    val, ctx, is_typeddict=is_typeddict, allow_unpack=allow_unpack
                )
                for val in value.vals
            ]
        )
    elif isinstance(value, AnnotatedValue):
        return _type_from_value(value.value, ctx)
    elif isinstance(value, _SubscriptedValue):
        return _type_from_subscripted_value(
            value.root,
            value.members,
            ctx,
            is_typeddict=is_typeddict,
            allow_unpack=allow_unpack,
        )
    elif isinstance(value, AnyValue):
        return value
    elif isinstance(value, SubclassValue) and value.exactly:
        return value.typ
    elif isinstance(value, TypedValue) and isinstance(value.typ, str):
        # Synthetic type
        return value
    else:
        ctx.show_error(f"Unrecognized annotation {value}")
        return AnyValue(AnySource.error)


def _type_from_subscripted_value(
    root: Optional[Value],
    members: Sequence[Value],
    ctx: Context,
    *,
    is_typeddict: bool = False,
    allow_unpack: bool = False,
) -> Value:
    if isinstance(root, GenericValue):
        if len(root.args) == len(members):
            return GenericValue(
                root.typ, [_type_from_value(member, ctx) for member in members]
            )
    if isinstance(root, _SubscriptedValue):
        root_type = _type_from_value(root, ctx)
        return _type_from_subscripted_value(root_type, members, ctx)
    elif isinstance(root, MultiValuedValue):
        return unite_values(
            *[
                _type_from_subscripted_value(
                    subval,
                    members,
                    ctx,
                    is_typeddict=is_typeddict,
                    allow_unpack=allow_unpack,
                )
                for subval in root.vals
            ]
        )
    if (
        isinstance(root, SubclassValue)
        and root.exactly
        and isinstance(root.typ, TypedValue)
    ):
        return GenericValue(
            root.typ.typ, [_type_from_value(elt, ctx) for elt in members]
        )

    if isinstance(root, TypedValue) and isinstance(root.typ, str):
        return GenericValue(root.typ, [_type_from_value(elt, ctx) for elt in members])

    if not isinstance(root, KnownValue):
        if root != AnyValue(AnySource.error):
            ctx.show_error(f"Cannot resolve subscripted annotation: {root}")
        return AnyValue(AnySource.error)
    root = root.val
    if root is typing.Union:
        return unite_values(*[_type_from_value(elt, ctx) for elt in members])
    elif is_typing_name(root, "Literal"):
        # Note that in Python 3.8, the way typing's internal cache works means that
        # Literal[1] and Literal[True] are cached to the same value, so if you use
        # both, you'll get whichever one was used first in later calls. There's nothing
        # we can do about that.
        if all(isinstance(elt, KnownValue) for elt in members):
            return unite_values(*members)
        else:
            ctx.show_error(f"Arguments to Literal[] must be literals, not {members}")
            return AnyValue(AnySource.error)
    elif root is typing.Tuple or root is tuple:
        if len(members) == 2 and members[1] == KnownValue(Ellipsis):
            return GenericValue(tuple, [_type_from_value(members[0], ctx)])
        elif len(members) == 1 and members[0] == KnownValue(()):
            return SequenceValue(tuple, [])
        else:
            return _make_sequence_value(
                tuple,
                [_type_from_value(arg, ctx, allow_unpack=True) for arg in members],
                ctx,
            )
    elif root is typing.Optional:
        if len(members) != 1:
            ctx.show_error("Optional[] takes only one argument")
            return AnyValue(AnySource.error)
        return unite_values(KnownValue(None), _type_from_value(members[0], ctx))
    elif root is typing.Type or root is type:
        if len(members) != 1:
            ctx.show_error("Type[] takes only one argument")
            return AnyValue(AnySource.error)
        argument = _type_from_value(members[0], ctx)
        return SubclassValue.make(argument)
    elif is_typing_name(root, "Annotated"):
        origin, *metadata = members
        return _make_annotated(_type_from_value(origin, ctx), metadata, ctx)
    elif is_typing_name(root, "TypeGuard"):
        if len(members) != 1:
            ctx.show_error("TypeGuard requires a single argument")
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeGuardExtension(_type_from_value(members[0], ctx))]
        )
    elif is_typing_name(root, "Required"):
        if not is_typeddict:
            ctx.show_error("Required[] used in unsupported context")
            return AnyValue(AnySource.error)
        if len(members) != 1:
            ctx.show_error("Required[] requires a single argument")
            return AnyValue(AnySource.error)
        return Pep655Value(True, _type_from_value(members[0], ctx))
    elif is_typing_name(root, "NotRequired"):
        if not is_typeddict:
            ctx.show_error("NotRequired[] used in unsupported context")
            return AnyValue(AnySource.error)
        if len(members) != 1:
            ctx.show_error("NotRequired[] requires a single argument")
            return AnyValue(AnySource.error)
        return Pep655Value(False, _type_from_value(members[0], ctx))
    elif is_typing_name(root, "Unpack"):
        if not allow_unpack:
            ctx.show_error("Unpack[] used in unsupported context")
            return AnyValue(AnySource.error)
        if len(members) != 1:
            ctx.show_error("Unpack requires a single argument")
            return AnyValue(AnySource.error)
        return UnpackedValue(_type_from_value(members[0], ctx))
    elif root is Callable or root is typing.Callable:
        if len(members) == 2:
            args, return_value = members
            return _make_callable_from_value(args, return_value, ctx)
        ctx.show_error("Callable requires exactly two arguments")
        return AnyValue(AnySource.error)
    elif root is AsynqCallable:
        if len(members) == 2:
            args, return_value = members
            return _make_callable_from_value(args, return_value, ctx, is_asynq=True)
        ctx.show_error("AsynqCallable requires exactly two arguments")
        return AnyValue(AnySource.error)
    elif typing_inspect.is_generic_type(root):
        origin = typing_inspect.get_origin(root)
        if origin is None:
            # On Python 3.9 at least, get_origin() of a class that inherits
            # from Generic[T] is None.
            origin = root
        origin = _maybe_get_extra(origin)
        return GenericValue(origin, [_type_from_value(elt, ctx) for elt in members])
    elif isinstance(root, type):
        return GenericValue(root, [_type_from_value(elt, ctx) for elt in members])
    else:
        # In Python 3.9, generics are implemented differently and typing.get_origin
        # can help.
        origin = get_origin(root)
        if isinstance(origin, type):
            return GenericValue(origin, [_type_from_value(elt, ctx) for elt in members])
        ctx.show_error(f"Unrecognized subscripted annotation: {root}")
        return AnyValue(AnySource.error)


def _maybe_get_extra(origin: type) -> Union[type, str]:
    # ContextManager is defined oddly and we lose the Protocol if we don't use
    # synthetic types.
    if any(origin is cls for cls in CONTEXT_MANAGER_TYPES):
        return "contextlib.AbstractContextManager"
    elif any(origin is cls for cls in ASYNC_CONTEXT_MANAGER_TYPES):
        return "contextlib.AbstractAsyncContextManager"
    else:
        # turn typing.List into list in some Python versions
        # compare https://github.com/ilevkivskyi/typing_inspect/issues/36
        extra_origin = getattr(origin, "__extra__", None)
        if extra_origin is not None:
            return extra_origin
        return origin


class _DefaultContext(Context):
    def __init__(
        self,
        visitor: "NameCheckVisitor",
        node: Optional[ast.AST],
        globals: Optional[Mapping[str, object]] = None,
        use_name_node_for_error: bool = False,
    ) -> None:
        super().__init__()
        self.visitor = visitor
        self.node = node
        self.globals = globals
        self.use_name_node_for_error = use_name_node_for_error

    def show_error(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.invalid_annotation,
        node: Optional[ast.AST] = None,
    ) -> None:
        if node is None:
            node = self.node
        if self.visitor is not None and node is not None:
            self.visitor.show_error(node, message, error_code)

    def get_name(self, node: ast.Name) -> Value:
        if self.visitor is not None:
            val, _ = self.visitor.resolve_name(
                node,
                error_node=node if self.use_name_node_for_error else self.node,
                suppress_errors=self.should_suppress_undefined_names,
            )
            return val
        elif self.globals is not None:
            if node.id in self.globals:
                return KnownValue(self.globals[node.id])
            elif hasattr(builtins, node.id):
                return KnownValue(getattr(builtins, node.id))
        if self.should_suppress_undefined_names:
            return AnyValue(AnySource.inference)
        self.show_error(
            f"Undefined name {node.id!r} used in annotation",
            ErrorCode.undefined_name,
            node=node,
        )
        return AnyValue(AnySource.error)


@dataclass(frozen=True)
class _SubscriptedValue(Value):
    root: Optional[Value]
    members: Tuple[Value, ...]


@dataclass
class Pep655Value(Value):
    required: bool
    value: Value


@dataclass
class DecoratorValue(Value):
    decorator: object
    args: Tuple[Value, ...]


class _Visitor(ast.NodeVisitor):
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def generic_visit(self, node: ast.AST) -> None:
        raise NotImplementedError(f"no visitor implemented for {node!r}")

    def visit_Name(self, node: ast.Name) -> Value:
        return self.ctx.get_name(node)

    def visit_Subscript(self, node: ast.Subscript) -> Value:
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if isinstance(index, SequenceValue):
            members = index.get_member_sequence()
            if members is None:
                # TODO support unpacking here
                return AnyValue(AnySource.inference)
            members = tuple(members)
        else:
            members = (index,)
        return _SubscriptedValue(value, members)

    def visit_Attribute(self, node: ast.Attribute) -> Optional[Value]:
        root_value = self.visit(node.value)
        return self.ctx.get_attribute(root_value, node)

    def visit_Tuple(self, node: ast.Tuple) -> Value:
        elts = [(False, self.visit(elt)) for elt in node.elts]
        return SequenceValue(tuple, elts)

    def visit_List(self, node: ast.List) -> Value:
        elts = [(False, self.visit(elt)) for elt in node.elts]
        return SequenceValue(list, elts)

    def visit_Index(self, node: ast.Index) -> Value:
        # class is unused in 3.9
        return self.visit(node.value)  # static analysis: ignore[undefined_attribute]

    def visit_Ellipsis(self, node: ast.Ellipsis) -> Value:
        return KnownValue(Ellipsis)

    def visit_Constant(self, node: ast.Constant) -> Value:
        return KnownValue(node.value)

    def visit_NameConstant(self, node: ast.NameConstant) -> Value:
        return KnownValue(node.value)

    def visit_Num(self, node: ast.Num) -> Value:
        return KnownValue(node.n)

    def visit_Str(self, node: ast.Str) -> Value:
        return KnownValue(node.s)

    def visit_Bytes(self, node: ast.Bytes) -> Value:
        return KnownValue(node.s)

    def visit_Expr(self, node: ast.Expr) -> Value:
        return self.visit(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> Optional[Value]:
        if isinstance(node.op, ast.BitOr):
            return _SubscriptedValue(
                KnownValue(Union), (self.visit(node.left), self.visit(node.right))
            )
        else:
            return None

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Optional[Value]:
        # Only int and float negation on literals are supported.
        if isinstance(node.op, ast.USub):
            operand = self.visit(node.operand)
            if isinstance(operand, KnownValue) and isinstance(
                operand.val, (int, float)
            ):
                return KnownValue(-operand.val)
        return None

    def visit_Call(self, node: ast.Call) -> Optional[Value]:
        func = self.visit(node.func)
        if not isinstance(func, KnownValue):
            return None
        if func.val == NewType:
            arg_values = [self.visit(arg) for arg in node.args]
            kwarg_values = [(kw.arg, self.visit(kw.value)) for kw in node.keywords]
            args = []
            kwargs = {}
            for arg_value in arg_values:
                if isinstance(arg_value, KnownValue):
                    args.append(arg_value.val)
                else:
                    return None
            for name, kwarg_value in kwarg_values:
                if name is None:
                    if isinstance(kwarg_value, KnownValue) and isinstance(
                        kwarg_value.val, dict
                    ):
                        kwargs.update(kwarg_value.val)
                    else:
                        return None
                else:
                    if isinstance(kwarg_value, KnownValue):
                        kwargs[name] = kwarg_value.val
                    else:
                        return None
            return KnownValue(func.val(*args, **kwargs))
        elif func.val == TypeVar:
            arg_values = [self.visit(arg) for arg in node.args]
            kwarg_values = [(kw.arg, self.visit(kw.value)) for kw in node.keywords]
            if not arg_values:
                self.ctx.show_error(
                    "TypeVar() requires at least one argument", node=node
                )
                return None
            name_val = arg_values[0]
            if not isinstance(name_val, KnownValue):
                self.ctx.show_error("TypeVar name must be a literal", node=node.args[0])
                return None
            constraints = []
            for arg_value in arg_values[1:]:
                constraints.append(_type_from_value(arg_value, self.ctx))
            bound = None
            for name, kwarg_value in kwarg_values:
                if name in ("covariant", "contravariant"):
                    continue
                elif name == "bound":
                    bound = _type_from_value(kwarg_value, self.ctx)
                else:
                    self.ctx.show_error(f"Unrecognized TypeVar kwarg {name}", node=node)
                    return None
            tv = TypeVar(name_val.val)
            return TypeVarValue(tv, bound, tuple(constraints))
        elif is_typing_name(func.val, "ParamSpec"):
            arg_values = [self.visit(arg) for arg in node.args]
            kwarg_values = [(kw.arg, self.visit(kw.value)) for kw in node.keywords]
            if not arg_values:
                self.ctx.show_error(
                    "ParamSpec() requires at least one argument", node=node
                )
                return None
            name_val = arg_values[0]
            if not isinstance(name_val, KnownValue):
                self.ctx.show_error(
                    "ParamSpec name must be a literal", node=node.args[0]
                )
                return None
            for name, _ in kwarg_values:
                self.ctx.show_error(f"Unrecognized ParamSpec kwarg {name}", node=node)
                return None
            tv = ParamSpec(name_val.val)
            return TypeVarValue(tv, is_paramspec=True)
        elif is_typing_name(func.val, "deprecated") or func.val is deprecated:
            if node.keywords:
                self.ctx.show_error(
                    "deprecated() does not accept keyword arguments", node=node
                )
                return None
            arg_values = tuple(self.visit(arg) for arg in node.args)
            return DecoratorValue(deprecated, arg_values)
        elif isinstance(func.val, type):
            if func.val is object:
                return AnyValue(AnySource.inference)
            return TypedValue(func.val)
        else:
            return None


def _value_of_origin_args(
    origin: object,
    args: Sequence[object],
    val: object,
    ctx: Context,
    *,
    is_typeddict: bool = False,
    allow_unpack: bool = False,
) -> Value:
    if origin is typing.Type or origin is type:
        if not args:
            return TypedValue(type)
        return SubclassValue.make(_type_from_runtime(args[0], ctx))
    elif origin is typing.Tuple or origin is tuple:
        if not args:
            return TypedValue(tuple)
        elif len(args) == 2 and args[1] is Ellipsis:
            return GenericValue(tuple, [_type_from_runtime(args[0], ctx)])
        elif len(args) == 1 and args[0] == ():
            return SequenceValue(tuple, [])
        else:
            args_vals = [
                _type_from_runtime(arg, ctx, allow_unpack=True) for arg in args
            ]
            return _make_sequence_value(tuple, args_vals, ctx)
    elif origin is typing.Union:
        return unite_values(*[_type_from_runtime(arg, ctx) for arg in args])
    elif origin is Callable or origin is typing.Callable:
        if len(args) == 0:
            return TypedValue(Callable)
        *arg_types, return_type = args
        if len(arg_types) == 1 and isinstance(arg_types[0], list):
            arg_types = arg_types[0]
        params = _callable_args_from_runtime(arg_types, "Callable", ctx)
        sig = Signature.make(params, _type_from_runtime(return_type, ctx))
        return CallableValue(sig)
    elif is_typing_name(origin, "Annotated"):
        origin, metadata = args
        # This should never happen
        if not isinstance(metadata, Iterable):
            ctx.show_error("Unexpected format in Annotated")
            return AnyValue(AnySource.error)
        return _make_annotated(
            _type_from_runtime(origin, ctx),
            [KnownValue(data) for data in metadata],
            ctx,
        )
    elif isinstance(origin, type):
        origin = _maybe_get_extra(origin)
        if args:
            args_vals = [_type_from_runtime(val, ctx) for val in args]
            return GenericValue(origin, args_vals)
        else:
            return _maybe_typed_value(origin)
    elif is_typing_name(origin, "TypeGuard"):
        if len(args) != 1:
            ctx.show_error("TypeGuard requires a single argument")
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeGuardExtension(_type_from_runtime(args[0], ctx))]
        )
    elif is_typing_name(origin, "Final"):
        if len(args) != 1:
            ctx.show_error("Final requires a single argument")
            return AnyValue(AnySource.error)
        # TODO(#160): properly support Final
        return _type_from_runtime(args[0], ctx)
    elif is_typing_name(origin, "ClassVar"):
        if len(args) != 1:
            ctx.show_error("ClassVar requires a single argument")
            return AnyValue(AnySource.error)
        return _type_from_runtime(args[0], ctx)
    elif is_typing_name(origin, "Required"):
        if not is_typeddict:
            ctx.show_error("Required[] used in unsupported context")
            return AnyValue(AnySource.error)
        if len(args) != 1:
            ctx.show_error("Required[] requires a single argument")
            return AnyValue(AnySource.error)
        return Pep655Value(True, _type_from_runtime(args[0], ctx))
    elif is_typing_name(origin, "NotRequired"):
        if not is_typeddict:
            ctx.show_error("NotRequired[] used in unsupported context")
            return AnyValue(AnySource.error)
        if len(args) != 1:
            ctx.show_error("NotRequired[] requires a single argument")
            return AnyValue(AnySource.error)
        return Pep655Value(False, _type_from_runtime(args[0], ctx))
    elif is_typing_name(origin, "Unpack"):
        if not allow_unpack:
            ctx.show_error("Invalid usage of Unpack")
            return AnyValue(AnySource.error)
        if len(args) != 1:
            ctx.show_error("Unpack requires a single argument")
            return AnyValue(AnySource.error)
        return UnpackedValue(_type_from_runtime(args[0], ctx))
    elif origin is None and isinstance(val, type):
        # This happens for SupportsInt in 3.7.
        return _maybe_typed_value(val)
    else:
        ctx.show_error(
            f"Unrecognized annotation {origin}[{', '.join(map(repr, args))}]"
        )
        return AnyValue(AnySource.error)


def _maybe_typed_value(val: Union[type, str]) -> Value:
    if val is type(None):
        return KnownValue(None)
    elif val is Hashable:
        return _HashableValue(val)
    return TypedValue(val)


def _make_sequence_value(
    typ: type, members: Sequence[Value], ctx: Context
) -> SequenceValue:
    pairs = []
    for val in members:
        if isinstance(val, UnpackedValue):
            elements = val.get_elements()
            if elements is None:
                ctx.show_error(f"Invalid usage of Unpack with {val}")
                elements = [(True, AnyValue(AnySource.error))]
            pairs += elements
        else:
            pairs.append((False, val))
    return SequenceValue(typ, pairs)


def _make_callable_from_value(
    args: Value, return_value: Value, ctx: Context, is_asynq: bool = False
) -> Value:
    return_annotation = _type_from_value(return_value, ctx)
    if args == KnownValue(Ellipsis):
        return CallableValue(
            Signature.make(
                [ELLIPSIS_PARAM], return_annotation=return_annotation, is_asynq=is_asynq
            )
        )
    elif isinstance(args, SequenceValue):
        params = []
        for i, (is_many, arg) in enumerate(args.members):
            annotation = _type_from_value(arg, ctx)
            if is_many:
                param = SigParameter(
                    f"@{i}",
                    kind=ParameterKind.VAR_POSITIONAL,
                    annotation=GenericValue(tuple, [annotation]),
                )
            else:
                param = SigParameter(
                    f"@{i}", kind=ParameterKind.POSITIONAL_ONLY, annotation=annotation
                )
            params.append(param)
        try:
            sig = Signature.make(params, return_annotation, is_asynq=is_asynq)
        except InvalidSignature as e:
            ctx.show_error(str(e))
            return AnyValue(AnySource.error)
        return CallableValue(sig)
    elif isinstance(args, KnownValue) and is_instance_of_typing_name(
        args.val, "ParamSpec"
    ):
        annotation = TypeVarValue(args.val, is_paramspec=True)
        params = [
            SigParameter("__P", kind=ParameterKind.PARAM_SPEC, annotation=annotation)
        ]
        sig = Signature.make(params, return_annotation, is_asynq=is_asynq)
        return CallableValue(sig)
    elif isinstance(args, TypeVarValue) and args.is_paramspec:
        params = [SigParameter("__P", kind=ParameterKind.PARAM_SPEC, annotation=args)]
        sig = Signature.make(params, return_annotation, is_asynq=is_asynq)
        return CallableValue(sig)
    elif (
        isinstance(args, _SubscriptedValue)
        and isinstance(args.root, KnownValue)
        and is_typing_name(args.root.val, "Concatenate")
    ):
        annotations = [_type_from_value(arg, ctx) for arg in args.members]
        params = [
            SigParameter(
                f"@{i}",
                kind=ParameterKind.PARAM_SPEC
                if i == len(annotations) - 1
                else ParameterKind.POSITIONAL_ONLY,
                annotation=annotation,
            )
            for i, annotation in enumerate(annotations)
        ]
        sig = Signature.make(params, return_annotation, is_asynq=is_asynq)
        return CallableValue(sig)
    else:
        ctx.show_error(f"Unrecognized Callable type argument {args}")
        return AnyValue(AnySource.error)


def _make_annotated(origin: Value, metadata: Sequence[Value], ctx: Context) -> Value:
    metadata = [_value_from_metadata(entry, ctx) for entry in metadata]
    return annotate_value(origin, metadata)


def _value_from_metadata(entry: Value, ctx: Context) -> Union[Value, Extension]:
    if isinstance(entry, KnownValue):
        if isinstance(entry.val, ParameterTypeGuard):
            return ParameterTypeGuardExtension(
                entry.val.varname, _type_from_runtime(entry.val.guarded_type, ctx)
            )
        elif isinstance(entry.val, NoReturnGuard):
            return NoReturnGuardExtension(
                entry.val.varname, _type_from_runtime(entry.val.guarded_type, ctx)
            )
        elif isinstance(entry.val, HasAttrGuard):
            return HasAttrGuardExtension(
                entry.val.varname,
                _type_from_runtime(entry.val.attribute_name, ctx),
                _type_from_runtime(entry.val.attribute_type, ctx),
            )
        elif isinstance(entry.val, CustomCheck):
            return CustomCheckExtension(entry.val)
    return entry


_CONTEXT_MANAGER_TYPES = {
    "typing.AsyncContextManager",
    "typing.ContextManager",
    "contextlib.AbstractContextManager",
    "contextlib.AbstractAsyncContextManager",
    *CONTEXT_MANAGER_TYPES,
    *ASYNC_CONTEXT_MANAGER_TYPES,
}


def is_context_manager_type(typ: Union[str, type]) -> bool:
    return typ in _CONTEXT_MANAGER_TYPES
