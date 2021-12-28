"""

Code for understanding function definitions.

"""
from abc import abstractmethod
import ast
import asyncio
import asynq
import enum
from dataclasses import dataclass
from itertools import zip_longest
from typing import Iterable, List, Optional, Sequence, Tuple, TypeVar, Union
from typing_extensions import Protocol

from .config import Config
from .error_code import ErrorCode
from .extensions import overload, real_overload
from .options import Options, PyObjectSequenceOption
from .node_visitor import ErrorContext
from .signature import SigParameter, ParameterKind, Signature
from .stacked_scopes import Composite
from .value import (
    CallableValue,
    CanAssignContext,
    TypedValue,
    Value,
    AnySource,
    AnyValue,
    KnownValue,
    GenericValue,
    SubclassValue,
    TypeVarValue,
    unite_values,
    CanAssignError,
)

FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda]
IMPLICIT_CLASSMETHODS = ("__init_subclass__", "__new__")


class AsyncFunctionKind(enum.Enum):
    non_async = 0
    normal = 1
    async_proxy = 2
    pure = 3


@dataclass(frozen=True)
class FunctionInfo:
    """Computed before visiting a function."""

    async_kind: AsyncFunctionKind
    is_classmethod: bool  # has @classmethod
    is_staticmethod: bool  # has @staticmethod
    is_decorated_coroutine: bool  # has @asyncio.coroutine
    is_overload: bool  # typing.overload or pyanalyze.extensions.overload
    is_abstractmethod: bool  # has @abstractmethod
    # a list of pairs of (decorator function, applied decorator function). These are different
    # for decorators that take arguments, like @asynq(): the first element will be the asynq
    # function and the second will be the result of calling asynq().
    decorators: List[Tuple[Value, Value]]


@dataclass(frozen=True)
class ParamInfo:
    param: SigParameter
    node: ast.AST
    is_self: bool = False


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

    def value_of_annotated_arg(self, __arg: ast.arg) -> Value:
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

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[object]:
        return list(fallback.ASYNQ_DECORATORS)


class AsyncProxyDecorators(PyObjectSequenceOption[object]):
    """Decorators that are equivalent to asynq.async_proxy."""

    default_value = [asynq.async_proxy]
    name = "async_proxy_decorators"

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[object]:
        return list(fallback.ASYNC_PROXY_DECORATORS)


class SafeDecoratorsForNestedFunctions(PyObjectSequenceOption[object]):
    """These decorators can safely be applied to nested functions."""

    name = "safe_decorators_for_nested_functions"
    default_value = [asynq.asynq, classmethod, staticmethod, asyncio.coroutine]

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[object]:
        return list(fallback.SAFE_DECORATORS_FOR_NESTED_FUNCTIONS)


def compute_function_info(decorator_list: List[ast.expr], ctx: Context) -> FunctionInfo:
    """Visits a function's decorator list."""
    async_kind = AsyncFunctionKind.non_async
    is_classmethod = False
    is_decorated_coroutine = False
    is_staticmethod = False
    is_overload = False
    is_abstractmethod = False
    decorators = []
    for decorator in decorator_list:
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
            decorators.append((callee, decorator_value))
        else:
            decorator_value = ctx.visit_expression(decorator)
            if decorator_value == KnownValue(classmethod):
                is_classmethod = True
            elif decorator_value == KnownValue(staticmethod):
                is_staticmethod = True
            elif decorator_value == KnownValue(asyncio.coroutine):
                is_decorated_coroutine = True
            elif decorator_value == KnownValue(
                real_overload
            ) or decorator_value == KnownValue(overload):
                is_overload = True
            elif decorator_value == KnownValue(abstractmethod):
                is_abstractmethod = True
            decorators.append((decorator_value, decorator_value))
    return FunctionInfo(
        async_kind=async_kind,
        is_decorated_coroutine=is_decorated_coroutine,
        is_classmethod=is_classmethod,
        is_staticmethod=is_staticmethod,
        is_abstractmethod=is_abstractmethod,
        is_overload=is_overload,
        decorators=decorators,
    )


def _visit_default(node: ast.AST, ctx: Context) -> Value:
    val = ctx.visit_expression(node)
    if val == KnownValue(...):
        return AnyValue(AnySource.unannotated)
    return val


def compute_parameters(
    node: FunctionNode,
    function_info: FunctionInfo,
    enclosing_class: Optional[TypedValue],
    ctx: Context,
    *,
    is_nested_in_class: bool = False,
) -> Sequence[ParamInfo]:
    """Visits and checks the arguments to a function."""
    defaults = [_visit_default(node, ctx) for node in node.args.defaults]
    kw_defaults = [
        None if kw_default is None else _visit_default(kw_default, ctx)
        for kw_default in node.args.kw_defaults
    ]

    posonly_args = getattr(node.args, "posonlyargs", [])
    num_without_defaults = len(node.args.args) + len(posonly_args) - len(defaults)
    defaults = [*[None] * num_without_defaults, *defaults, *kw_defaults]
    args: List[Tuple[ParameterKind, ast.arg]] = (
        [(ParameterKind.POSITIONAL_ONLY, arg) for arg in posonly_args]
        + [(ParameterKind.POSITIONAL_OR_KEYWORD, arg) for arg in node.args.args]
        + [(ParameterKind.KEYWORD_ONLY, arg) for arg in node.args.kwonlyargs]
    )
    if node.args.vararg is not None:
        args.append((ParameterKind.VAR_POSITIONAL, node.args.vararg))
    if node.args.kwarg is not None:
        args.append((ParameterKind.VAR_KEYWORD, node.args.kwarg))
    params = []
    tv_index = 1

    for idx, ((kind, arg), default) in enumerate(zip_longest(args, defaults)):
        is_self = (
            idx == 0
            and enclosing_class is not None
            and not function_info.is_staticmethod
            and not isinstance(node, ast.Lambda)
        )
        if arg.annotation is not None:
            value = ctx.value_of_annotated_arg(arg)
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
            if (
                function_info.is_classmethod
                or getattr(node, "name", None) in IMPLICIT_CLASSMETHODS
            ):
                value = SubclassValue(enclosing_class)
            else:
                # normal method
                value = enclosing_class
        else:
            # This is meant to exclude methods in nested classes. It's a bit too
            # conservative for cases such as a function nested in a method nested in a
            # class nested in a function.
            if not isinstance(node, ast.Lambda) and not (
                idx == 0 and not function_info.is_staticmethod and is_nested_in_class
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

        if kind is ParameterKind.VAR_POSITIONAL:
            value = GenericValue(tuple, [value])
        elif kind is ParameterKind.VAR_KEYWORD:
            value = GenericValue(dict, [TypedValue(str), value])

        param = SigParameter(arg.arg, kind, default, value)
        info = ParamInfo(param, arg, is_self)
        params.append(info)
    return params


def compute_value_of_function(
    node: FunctionNode, info: FunctionInfo, result: FunctionResult, ctx: Context
) -> Value:
    sig = Signature.make(
        result.parameters,
        result.return_value,
        has_return_annotation=result.has_return_annotation,
    )
    val = CallableValue(sig)
    for unapplied, decorator in reversed(info.decorators):
        allow_call = isinstance(
            unapplied, KnownValue
        ) and SafeDecoratorsForNestedFunctions.contains(unapplied.val, ctx.options)
        val = ctx.check_call(node, decorator, [Composite(val)], allow_call=allow_call)
    return val
