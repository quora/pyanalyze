"""

Code for understanding function definitions.

"""

import ast
import asyncio
import collections.abc
import enum
import sys
import types
from dataclasses import dataclass, replace
from itertools import zip_longest
from typing import Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import asynq
from typing_extensions import Protocol

from .error_code import ErrorCode
from .node_visitor import ErrorContext
from .options import Options, PyObjectSequenceOption
from .signature import ParameterKind, Signature, SigParameter
from .stacked_scopes import Composite
from .value import (
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    GenericValue,
    KnownValue,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    SubclassValue,
    TypedValue,
    TypeVarValue,
    UnpackedValue,
    Value,
    get_tv_map,
    is_async_iterable,
    is_iterable,
    make_coro_type,
    unite_values,
)

FunctionDefNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]
FunctionNode = Union[FunctionDefNode, ast.Lambda]
IMPLICIT_CLASSMETHODS = ("__init_subclass__", "__new__")

YieldT = TypeVar("YieldT")
SendT = TypeVar("SendT")
ReturnT = TypeVar("ReturnT")
GeneratorValue = GenericValue(
    collections.abc.Generator,
    [TypeVarValue(YieldT), TypeVarValue(SendT), TypeVarValue(ReturnT)],
)
AsyncGeneratorValue = GenericValue(
    collections.abc.AsyncGenerator, [TypeVarValue(YieldT), TypeVarValue(SendT)]
)


class AsyncFunctionKind(enum.Enum):
    non_async = 0
    normal = 1
    async_proxy = 2
    pure = 3


@dataclass(frozen=True)
class ParamInfo:
    param: SigParameter
    node: ast.AST
    is_self: bool = False


@dataclass(frozen=True)
class FunctionInfo:
    """Computed before visiting a function."""

    async_kind: AsyncFunctionKind
    is_classmethod: bool  # has @classmethod
    is_staticmethod: bool  # has @staticmethod
    is_decorated_coroutine: bool  # has @asyncio.coroutine
    is_overload: bool  # typing.overload or pyanalyze.extensions.overload
    is_override: bool  # @typing.override
    is_evaluated: bool  # @pyanalyze.extensions.evaluated
    is_abstractmethod: bool  # has @abstractmethod
    is_instancemethod: bool  # is an instance method
    # a list of tuples of (decorator function, applied decorator function, AST node). These are
    # different for decorators that take arguments, like @asynq(): the first element will be the
    # asynq function and the second will be the result of calling asynq().
    decorators: List[Tuple[Value, Value, ast.AST]]
    node: FunctionNode
    params: Sequence[ParamInfo]
    return_annotation: Optional[Value]
    potential_function: Optional[object]
    type_params: Sequence[TypeVarValue]

    def get_generator_yield_type(self, ctx: CanAssignContext) -> Value:
        if self.return_annotation is None:
            return AnyValue(AnySource.unannotated)
        if isinstance(self.node, ast.AsyncFunctionDef):
            iterable_val = is_async_iterable(self.return_annotation, ctx)
            if isinstance(iterable_val, CanAssignError):
                return AnyValue(AnySource.error)
            return iterable_val
        else:
            iterable_val = is_iterable(self.return_annotation, ctx)
            if isinstance(iterable_val, CanAssignError):
                return AnyValue(AnySource.error)
            return iterable_val

    def get_generator_send_type(self, ctx: CanAssignContext) -> Value:
        if self.return_annotation is None:
            return AnyValue(AnySource.unannotated)
        if isinstance(self.node, ast.AsyncFunctionDef):
            tv_map = get_tv_map(AsyncGeneratorValue, self.return_annotation, ctx)
            if not isinstance(tv_map, CanAssignError):
                return tv_map.get(SendT, AnyValue(AnySource.generic_argument))
            # If the return annotation is a non-Generator Iterable, assume the send
            # type is None.
            iterable_val = is_async_iterable(self.return_annotation, ctx)
            if isinstance(iterable_val, CanAssignError):
                return AnyValue(AnySource.error)
            return KnownValue(None)
        else:
            tv_map = get_tv_map(GeneratorValue, self.return_annotation, ctx)
            if not isinstance(tv_map, CanAssignError):
                return tv_map.get(SendT, AnyValue(AnySource.generic_argument))
            # If the return annotation is a non-Generator Iterable, assume the send
            # type is None.
            iterable_val = is_iterable(self.return_annotation, ctx)
            if isinstance(iterable_val, CanAssignError):
                return AnyValue(AnySource.error)
            return KnownValue(None)

    def get_generator_return_type(self, ctx: CanAssignContext) -> Value:
        if self.return_annotation is None:
            return AnyValue(AnySource.unannotated)
        tv_map = get_tv_map(GeneratorValue, self.return_annotation, ctx)
        if not isinstance(tv_map, CanAssignError):
            return tv_map.get(ReturnT, AnyValue(AnySource.generic_argument))
        # If the return annotation is a non-Generator Iterable, assume the return
        # type is None.
        iterable_val = is_iterable(self.return_annotation, ctx)
        if isinstance(iterable_val, CanAssignError):
            return AnyValue(AnySource.error)
        return KnownValue(None)


@dataclass
class FunctionResult:
    """Computed after visiting a function."""

    return_value: Value = AnyValue(AnySource.inference)
    parameters: Sequence[SigParameter] = ()
    has_return: bool = False
    is_generator: bool = False
    has_return_annotation: bool = False


class Context(ErrorContext, CanAssignContext, Protocol):
    options: Options

    def visit_expression(self, __node: ast.AST) -> Value:
        raise NotImplementedError

    def value_of_annotation(
        self, __node: ast.expr, *, allow_unpack: bool = False
    ) -> Value:
        raise NotImplementedError

    def check_call(
        self,
        node: ast.AST,
        callee: Value,
        args: Iterable[Composite],
        *,
        allow_call: bool = False,
    ) -> Value:
        raise NotImplementedError


class AsynqDecorators(PyObjectSequenceOption[object]):
    """Decorators that are equivalent to asynq.asynq."""

    default_value = [asynq.asynq]
    name = "asynq_decorators"


class AsyncProxyDecorators(PyObjectSequenceOption[object]):
    """Decorators that are equivalent to asynq.async_proxy."""

    default_value = [asynq.async_proxy]
    name = "async_proxy_decorators"


_safe_decorators = [asynq.asynq, classmethod, staticmethod]
if sys.version_info < (3, 11):
    # static analysis: ignore[undefined_attribute]
    _safe_decorators.append(asyncio.coroutine)


class SafeDecoratorsForNestedFunctions(PyObjectSequenceOption[object]):
    """These decorators can safely be applied to nested functions."""

    name = "safe_decorators_for_nested_functions"
    default_value = _safe_decorators


def _visit_default(node: ast.AST, ctx: Context) -> Value:
    val = ctx.visit_expression(node)
    if val == KnownValue(...):
        return AnyValue(AnySource.unannotated)
    return val


def compute_parameters(
    node: FunctionNode,
    enclosing_class: Optional[TypedValue],
    ctx: Context,
    *,
    is_nested_in_class: bool = False,
    is_staticmethod: bool = False,
    is_classmethod: bool = False,
) -> Sequence[ParamInfo]:
    """Visits and checks the arguments to a function."""
    defaults = [_visit_default(node, ctx) for node in node.args.defaults]
    kw_defaults = [
        None if kw_default is None else _visit_default(kw_default, ctx)
        for kw_default in node.args.kw_defaults
    ]

    posonly_args = getattr(node.args, "posonlyargs", [])
    num_without_defaults = len(node.args.args) + len(posonly_args) - len(defaults)
    vararg_defaults = [None] if node.args.vararg is not None else []
    defaults = [
        *[None] * num_without_defaults,
        *defaults,
        *vararg_defaults,
        *kw_defaults,
    ]
    args: List[Tuple[ParameterKind, ast.arg]] = [
        (ParameterKind.POSITIONAL_ONLY, arg) for arg in posonly_args
    ] + [(ParameterKind.POSITIONAL_OR_KEYWORD, arg) for arg in node.args.args]
    if node.args.vararg is not None:
        args.append((ParameterKind.VAR_POSITIONAL, node.args.vararg))
    args += [(ParameterKind.KEYWORD_ONLY, arg) for arg in node.args.kwonlyargs]
    if node.args.kwarg is not None:
        args.append((ParameterKind.VAR_KEYWORD, node.args.kwarg))
    params = []
    tv_index = 1

    seen_paramspec_args: Optional[Tuple[ast.arg, ParamSpecArgsValue]] = None
    for idx, (param, default) in enumerate(zip_longest(args, defaults)):
        assert param is not None, "must have more args than defaults"
        (kind, arg) = param
        is_self = (
            idx == 0
            and enclosing_class is not None
            and not is_staticmethod
            and not isinstance(node, ast.Lambda)
        )
        if arg.annotation is not None:
            value = ctx.value_of_annotation(
                arg.annotation, allow_unpack=kind.allow_unpack()
            )
            if default is not None:
                tv_map = value.can_assign(default, ctx)
                if isinstance(tv_map, CanAssignError):
                    ctx.show_error(
                        arg,
                        f"Default value for argument {arg.arg} incompatible"
                        f" with declared type {value}",
                        error_code=ErrorCode.incompatible_default,
                        detail=tv_map.display(),
                    )
        elif is_self:
            assert enclosing_class is not None
            if is_classmethod or getattr(node, "name", None) in IMPLICIT_CLASSMETHODS:
                value = SubclassValue(enclosing_class)
            else:
                # normal method
                value = enclosing_class
        else:
            # This is meant to exclude methods in nested classes. It's a bit too
            # conservative for cases such as a function nested in a method nested in a
            # class nested in a function.
            if not isinstance(node, ast.Lambda) and not (
                idx == 0 and not is_staticmethod and is_nested_in_class
            ):
                ctx.show_error(
                    arg,
                    f"Missing type annotation for parameter {arg.arg}",
                    error_code=ErrorCode.missing_parameter_annotation,
                )
            if isinstance(node, ast.Lambda):
                value = TypeVarValue(TypeVar(f"T{tv_index}"))
                tv_index += 1
            else:
                value = AnyValue(AnySource.unannotated)
            if default is not None:
                value = unite_values(value, default)

        value = translate_vararg_type(kind, value, ctx, error_ctx=ctx, node=arg)
        if isinstance(value, ParamSpecArgsValue):
            if kind is ParameterKind.VAR_POSITIONAL:
                seen_paramspec_args = (arg, value)
            else:
                ctx.show_error(
                    arg,
                    f"ParamSpec.args must be used on *args, not {arg.arg}",
                    error_code=ErrorCode.invalid_annotation,
                )
        elif isinstance(value, ParamSpecKwargsValue):
            if kind is ParameterKind.VAR_KEYWORD:
                if seen_paramspec_args is not None:
                    _, ps_args = seen_paramspec_args
                    if ps_args.param_spec is not value.param_spec:
                        ctx.show_error(
                            arg,
                            "The same ParamSpec must be used on *args and **kwargs",
                            error_code=ErrorCode.invalid_annotation,
                        )
                else:
                    ctx.show_error(
                        arg,
                        "ParamSpec.kwargs must be used together with ParamSpec.args",
                        error_code=ErrorCode.invalid_annotation,
                    )
            else:
                ctx.show_error(
                    arg,
                    f"ParamSpec.kwargs must be used on **kwargs, not {arg.arg}",
                    error_code=ErrorCode.invalid_annotation,
                )

        param = SigParameter(arg.arg, kind, default, value)
        info = ParamInfo(param, arg, is_self)
        params.append(info)
    return params


def translate_vararg_type(
    kind: ParameterKind,
    typ: Value,
    can_assign_ctx: CanAssignContext,
    *,
    error_ctx: Optional[ErrorContext] = None,
    node: Optional[ast.AST] = None,
) -> Value:
    if kind is ParameterKind.VAR_POSITIONAL:
        if isinstance(typ, UnpackedValue):
            if not TypedValue(tuple).is_assignable(typ.value, can_assign_ctx):
                if error_ctx is not None and node is not None:
                    error_ctx.show_error(
                        node,
                        "Expected tuple type inside Unpack[]",
                        error_code=ErrorCode.invalid_annotation,
                    )
                return AnyValue(AnySource.error)
            return typ.value
        elif isinstance(typ, ParamSpecArgsValue):
            return typ
        else:
            return GenericValue(tuple, [typ])
    elif kind is ParameterKind.VAR_KEYWORD:
        if isinstance(typ, UnpackedValue):
            if not TypedValue(dict).is_assignable(typ.value, can_assign_ctx):
                if error_ctx is not None and node is not None:
                    error_ctx.show_error(
                        node,
                        "Expected dict type inside Unpack[]",
                        error_code=ErrorCode.invalid_annotation,
                    )
                return AnyValue(AnySource.error)
            return typ.value
        elif isinstance(typ, ParamSpecKwargsValue):
            return typ
        else:
            return GenericValue(dict, [TypedValue(str), typ])
    return typ


@dataclass
class IsGeneratorVisitor(ast.NodeVisitor):
    """Determine whether an async function is a generator.

    This is important because the return type of async generators
    should not be wrapped in Coroutine.

    We avoid recursing into nested functions, which is why we can't
    just use ast.walk.

    We do not need to check for yield from because it is illegal
    in async generators. We also skip checking nested comprehensions,
    because we error anyway if there is a yield within a comprehension.

    """

    is_generator: bool = False

    def visit_Yield(self, node: ast.Yield) -> None:
        self.is_generator = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass


def compute_value_of_function(
    info: FunctionInfo, ctx: Context, *, result: Optional[Value] = None
) -> Value:
    if result is None:
        result = info.return_annotation
    if result is None:
        result = AnyValue(AnySource.unannotated)
    if isinstance(info.node, ast.AsyncFunctionDef):
        visitor = IsGeneratorVisitor()
        for line in info.node.body:
            visitor.visit(line)
            if visitor.is_generator:
                break
        if not visitor.is_generator:
            result = make_coro_type(result)
    sig = Signature.make(
        [param_info.param for param_info in info.params],
        result,
        has_return_annotation=info.return_annotation is not None,
    )
    val = CallableValue(sig, types.FunctionType)
    for unapplied, decorator, node in reversed(info.decorators):
        # Special case asynq.asynq until we can create the type automatically
        if unapplied == KnownValue(asynq.asynq) and isinstance(val, CallableValue):
            sig = replace(val.signature, is_asynq=True)
            val = CallableValue(sig, val.typ)
            continue
        allow_call = isinstance(
            unapplied, KnownValue
        ) and SafeDecoratorsForNestedFunctions.contains(unapplied.val, ctx.options)
        val = ctx.check_call(node, decorator, [Composite(val)], allow_call=allow_call)
    return val
