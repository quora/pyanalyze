"""

Code for understanding function definitions.

"""
import ast
import asyncio
import collections.abc
import enum
import sys
import types
from abc import abstractmethod
from dataclasses import dataclass, replace
from itertools import zip_longest
from typing import Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import asynq
from typing_extensions import Protocol

from .error_code import ErrorCode
from .extensions import evaluated, overload, real_overload
from .node_visitor import ErrorContext
from .options import Options, PyObjectSequenceOption
from .signature import ParameterKind, Signature, SigParameter
from .stacked_scopes import Composite
from .typevar import resolve_bounds_map
from .value import (
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    GenericValue,
    get_tv_map,
    KnownValue,
    make_coro_type,
    SubclassValue,
    TypedValue,
    TypeVarValue,
    unite_values,
    UnpackedValue,
    Value,
)

FunctionDefNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]
FunctionNode = Union[FunctionDefNode, ast.Lambda]
IMPLICIT_CLASSMETHODS = ("__init_subclass__", "__new__")

YieldT = TypeVar("YieldT")
SendT = TypeVar("SendT")
ReturnT = TypeVar("ReturnT")
IterableValue = GenericValue(collections.abc.Iterable, [TypeVarValue(YieldT)])
GeneratorValue = GenericValue(
    collections.abc.Generator,
    [TypeVarValue(YieldT), TypeVarValue(SendT), TypeVarValue(ReturnT)],
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
    is_evaluated: bool  # @pyanalyze.extensions.evaluated
    is_abstractmethod: bool  # has @abstractmethod
    # a list of tuples of (decorator function, applied decorator function, AST node). These are
    # different for decorators that take arguments, like @asynq(): the first element will be the
    # asynq function and the second will be the result of calling asynq().
    decorators: List[Tuple[Value, Value, ast.AST]]
    node: FunctionNode
    params: Sequence[ParamInfo]
    return_annotation: Optional[Value]
    potential_function: Optional[object]

    def get_generator_yield_type(self, ctx: CanAssignContext) -> Value:
        if self.return_annotation is None:
            return AnyValue(AnySource.unannotated)
        can_assign = IterableValue.can_assign(self.return_annotation, ctx)
        if isinstance(can_assign, CanAssignError):
            return AnyValue(AnySource.error)
        tv_map, _ = resolve_bounds_map(can_assign, ctx)
        return tv_map.get(YieldT, AnyValue(AnySource.generic_argument))

    def get_generator_send_type(self, ctx: CanAssignContext) -> Value:
        if self.return_annotation is None:
            return AnyValue(AnySource.unannotated)
        tv_map = get_tv_map(GeneratorValue, self.return_annotation, ctx)
        if not isinstance(tv_map, CanAssignError):
            return tv_map.get(SendT, AnyValue(AnySource.generic_argument))
        # If the return annotation is a non-Generator Iterable, assume the send
        # type is None.
        can_assign = IterableValue.can_assign(self.return_annotation, ctx)
        if isinstance(can_assign, CanAssignError):
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
        can_assign = IterableValue.can_assign(self.return_annotation, ctx)
        if isinstance(can_assign, CanAssignError):
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


def compute_function_info(
    node: FunctionNode,
    ctx: Context,
    *,
    is_nested_in_class: bool = False,
    enclosing_class: Optional[TypedValue] = None,
    potential_function: Optional[object] = None,
) -> FunctionInfo:
    """Visits a function's decorator list."""
    async_kind = AsyncFunctionKind.non_async
    is_classmethod = False
    is_decorated_coroutine = False
    is_staticmethod = False
    is_overload = False
    is_abstractmethod = False
    is_evaluated = False
    decorators = []
    for decorator in [] if isinstance(node, ast.Lambda) else node.decorator_list:
        # We have to descend into the Call node because the result of
        # asynq.asynq() is a one-off function that we can't test against.
        # This means that the decorator will be visited more than once, which seems OK.
        if isinstance(decorator, ast.Call):
            decorator_value = ctx.visit_expression(decorator)
            callee = ctx.visit_expression(decorator.func)
            if isinstance(callee, KnownValue):
                if AsynqDecorators.contains(callee.val, ctx.options):
                    if any(kw.arg == "pure" for kw in decorator.keywords):
                        async_kind = AsyncFunctionKind.pure
                    else:
                        async_kind = AsyncFunctionKind.normal
                elif AsyncProxyDecorators.contains(callee.val, ctx.options):
                    # @async_proxy(pure=True) is a noop, so don't treat it specially
                    if not any(kw.arg == "pure" for kw in decorator.keywords):
                        async_kind = AsyncFunctionKind.async_proxy
            decorators.append((callee, decorator_value, decorator))
        else:
            decorator_value = ctx.visit_expression(decorator)
            if decorator_value == KnownValue(classmethod):
                is_classmethod = True
            elif decorator_value == KnownValue(staticmethod):
                is_staticmethod = True
            elif sys.version_info < (3, 11) and decorator_value == KnownValue(
                asyncio.coroutine  # static analysis: ignore[undefined_attribute]
            ):
                is_decorated_coroutine = True
            elif decorator_value == KnownValue(
                real_overload
            ) or decorator_value == KnownValue(overload):
                is_overload = True
            elif decorator_value == KnownValue(abstractmethod):
                is_abstractmethod = True
            elif decorator_value == KnownValue(evaluated):
                is_evaluated = True
            decorators.append((decorator_value, decorator_value, decorator))
    params = compute_parameters(
        node,
        enclosing_class,
        ctx,
        is_nested_in_class=is_nested_in_class,
        is_classmethod=is_classmethod,
        is_staticmethod=is_staticmethod,
    )
    if isinstance(node, ast.Lambda) or node.returns is None:
        return_annotation = None
    else:
        return_annotation = ctx.value_of_annotation(node.returns)
    return FunctionInfo(
        async_kind=async_kind,
        is_decorated_coroutine=is_decorated_coroutine,
        is_classmethod=is_classmethod,
        is_staticmethod=is_staticmethod,
        is_abstractmethod=is_abstractmethod,
        is_overload=is_overload,
        is_evaluated=is_evaluated,
        decorators=decorators,
        node=node,
        params=params,
        return_annotation=return_annotation,
        potential_function=potential_function,
    )


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
                    node,
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
