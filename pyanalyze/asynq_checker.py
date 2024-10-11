"""

Implementation of test_scope's asynq checks.

"""

import ast
import contextlib
import inspect
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import asynq
import qcore

from .error_code import ErrorCode
from .functions import AsyncFunctionKind
from .options import Options, PyObjectSequenceOption, StringSequenceOption
from .safe import safe_getattr, safe_hasattr
from .value import AnnotatedValue, KnownValue, TypedValue, UnboundMethodValue, Value


class ClassesCheckedForAsynq(PyObjectSequenceOption[type]):
    """Normally, asynq calls to asynq functions are only enforced in functions that are already
    asynq. In subclasses of classes listed here, all asynq functions must be called asynq.
    """

    name = "classes_checked_for_asynq"


class NonAsynqModules(StringSequenceOption):
    """We ignore _async methods in these modules."""

    name = "non_asynq_modules"
    default_value = ["multiprocessing"]


class MethodsNotCheckedForAsynq(StringSequenceOption):
    """Async batching in these component methods isn't checked even when they exist on a class in
    ClassesCheckedForAsynq."""

    name = "methods_not_checked_for_asynq"


@dataclass
class AsynqChecker:
    options: Options
    module: Optional[types.ModuleType]
    on_error: Callable[..., Any]
    log: Callable[..., Any]
    replace_node: Callable[..., Any]
    current_func_name: Optional[str] = field(init=False, default=None)
    current_class: Optional[type] = field(init=False, default=None)
    current_async_kind: AsyncFunctionKind = field(
        init=False, default=AsyncFunctionKind.non_async
    )
    is_classmethod: bool = field(init=False, default=False)

    # Functions called from test_scope itself

    @contextlib.contextmanager
    def set_func_name(
        self,
        name: str,
        async_kind: AsyncFunctionKind = AsyncFunctionKind.non_async,
        is_classmethod: bool = False,
    ) -> Iterator[None]:
        """Sets the current function name for async data collection."""
        # Override current_func_name only if this is the outermost function, so that data access
        # within nested functions is attributed to the outer function. However, for async inner
        # functions, check batching within the function separately.
        with (
            qcore.override(self, "current_async_kind", async_kind),
            qcore.override(self, "is_classmethod", is_classmethod),
        ):
            if (
                self.current_func_name is None
                or async_kind != AsyncFunctionKind.non_async
            ):
                with qcore.override(self, "current_func_name", name):
                    yield
            else:
                yield

    def check_call(self, value: Value, node: ast.Call) -> None:
        if not self.should_perform_async_checks():
            return
        if is_impure_async_fn(value):
            func_node = ast.Attribute(value=node.func, attr="asynq")
            call_node = replace_func_on_call_node(node, func_node)
            replacement_node = ast.Yield(value=call_node)
            self._show_impure_async_error(
                node,
                replacement_call=get_pure_async_equivalent(value),
                replacement_node=replacement_node,
            )
        elif (
            isinstance(value, UnboundMethodValue)
            and value.secondary_attr_name is None
            and isinstance(value.composite.value, TypedValue)
        ):
            inner_type = value.composite.value.typ
            if not safe_hasattr(inner_type, value.attr_name + "_async"):
                return
            module = safe_getattr(inner_type, "__module__", None)
            if isinstance(module, str) and module.split(".")[
                0
            ] in self.options.get_value_for(NonAsynqModules):
                return
            if isinstance(node.func, ast.Attribute):
                func_node = ast.Attribute(
                    value=node.func.value, attr=value.attr_name + "_async"
                )
                call_node = replace_func_on_call_node(node, func_node)
                replacement_node = ast.Yield(value=call_node)
            else:
                replacement_node = None
            self._show_impure_async_error(
                node,
                replacement_call=f"{_stringify_obj(inner_type)}.{value.attr_name}_async",
                replacement_node=replacement_node,
            )

    def should_perform_async_checks(self) -> bool:
        if self.current_async_kind in (
            AsyncFunctionKind.normal,
            AsyncFunctionKind.pure,
        ):
            return True
        if self.current_class is None or self.is_classmethod:
            return False
        if self.current_func_name in self.options.get_value_for(
            MethodsNotCheckedForAsynq
        ):
            return False
        return self.should_check_class_for_async(self.current_class)

    def should_check_class_for_async(self, cls: type) -> bool:
        """Returns whether we should perform async checks on all methods on this class."""
        for base_cls in self.options.get_value_for(ClassesCheckedForAsynq):
            try:
                if issubclass(cls, base_cls):
                    return True
            except TypeError:
                pass  # old-style class or something
        return False

    def _show_impure_async_error(
        self,
        node: ast.AST,
        replacement_call: Optional[str] = None,
        replacement_node: Optional[ast.AST] = None,
    ) -> None:
        if replacement_call is None:
            message = "impure async call (you should yield it)"
        else:
            message = f"impure async call (you should yield {replacement_call})"
        if self.current_async_kind not in (
            AsyncFunctionKind.normal,
            AsyncFunctionKind.pure,
        ):
            # we're in a component method that is checked for async
            message += " and make this method async"
        if replacement_node is None:
            replacement = None
        else:
            replacement = self.replace_node(node, replacement_node)
        self.on_error(
            node, message, ErrorCode.impure_async_call, replacement=replacement
        )


def is_impure_async_fn(value: Value) -> bool:
    """Returns whether the given Value is an impure async call.

    This can be used to detect places where async functions are called synchronously.

    """
    if isinstance(value, KnownValue):
        return asynq.is_async_fn(value.val) and not asynq.is_pure_async_fn(value.val)
    elif isinstance(value, UnboundMethodValue):
        method = value.get_method()
        if method is None:
            return False
        return asynq.is_async_fn(method) and not asynq.is_pure_async_fn(method)
    return False


def get_pure_async_equivalent(value: Value) -> str:
    """Returns the pure-async equivalent of an async function."""
    assert is_impure_async_fn(value), f"{value} is not an impure async function"
    if isinstance(value, KnownValue):
        return f"{_stringify_obj(value.val)}.asynq"
    elif isinstance(value, UnboundMethodValue):
        return _stringify_async_fn(
            UnboundMethodValue(value.attr_name, value.composite, "asynq")
        )
    else:
        assert False, f"cannot get pure async equivalent of {value}"


def _stringify_async_fn(value: Value) -> str:
    if isinstance(value, KnownValue):
        return _stringify_obj(value.val)
    elif isinstance(value, UnboundMethodValue):
        typ = _stringify_async_fn(value.composite.value)
        ret = f"{typ}.{value.attr_name}"
        if value.secondary_attr_name is not None:
            ret += f".{value.secondary_attr_name}"
        return ret
    elif isinstance(value, TypedValue):
        return _stringify_obj(value.typ)
    elif isinstance(value, AnnotatedValue):
        return _stringify_async_fn(value.value)
    else:
        return str(value)


def _stringify_obj(obj: Any) -> str:
    # covers .asynq on async methods
    if (inspect.isbuiltin(obj) and obj.__self__ is not None) or isinstance(
        obj, types.MethodType
    ):
        return f"{_stringify_obj(obj.__self__)}.{obj.__name__}"
    elif hasattr(obj, "decorator") and hasattr(obj, "instance"):
        if hasattr(obj.instance, "__name__"):
            cls = obj.instance
        else:
            cls = type(obj.instance)
        return f"{_stringify_obj(cls)}.{obj.decorator.fn.__name__}"
    elif isinstance(obj, super):
        # self might not always be correct, but it's close enough
        return f"super({_stringify_obj(obj.__self_class__)}, self)"
    else:
        return f"{obj.__module__}.{obj.__name__}"


def replace_func_on_call_node(node: ast.Call, new_func: ast.expr) -> ast.Call:
    return ast.Call(func=new_func, args=node.args, keywords=node.keywords)
