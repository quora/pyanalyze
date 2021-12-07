"""

Implementation of test_scope's asynq checks.

"""

import ast
import asynq
import contextlib
from dataclasses import dataclass, field
import enum
import qcore
import inspect
import types
from typing import Sequence, Any, Callable, List, Optional, Iterable

from .config import Config
from .error_code import ErrorCode
from .safe import safe_getattr, safe_hasattr
from .stacked_scopes import Composite
from .value import AnnotatedValue, Value, KnownValue, TypedValue, UnboundMethodValue


class AsyncFunctionKind(enum.Enum):
    non_async = 0
    normal = 1
    async_proxy = 2
    pure = 3


@dataclass(frozen=True)
class FunctionInfo:
    async_kind: AsyncFunctionKind
    is_classmethod: bool  # has @classmethod
    is_staticmethod: bool  # has @staticmethod
    is_decorated_coroutine: bool  # has @asyncio.coroutine
    is_overload: bool  # typing.overload or pyanalyze.extensions.overload
    # a list of pairs of (decorator function, applied decorator function). These are different
    # for decorators that take arguments, like @asynq(): the first element will be the asynq
    # function and the second will be the result of calling asynq().
    decorators: List[Any]


@dataclass
class AsynqChecker:
    config: Config
    module: Optional[types.ModuleType]
    lines: Sequence[str]
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
    ) -> Iterable[None]:
        """Sets the current function name for async data collection."""
        # Override current_func_name only if this is the outermost function, so that data access
        # within nested functions is attributed to the outer function. However, for async inner
        # functions, check batching within the function separately.
        with qcore.override(self, "current_async_kind", async_kind), qcore.override(
            self, "is_classmethod", is_classmethod
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
            if (
                isinstance(module, str)
                and module.split(".")[0] in self.config.NON_ASYNQ_MODULES
            ):
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
                replacement_call="%s.%s_async"
                % (_stringify_obj(inner_type), value.attr_name),
                replacement_node=replacement_node,
            )

    def record_attribute_access(
        self, root_value: Value, attr_name: str, node: ast.Attribute
    ) -> None:
        """Records that the given attribute of root_value was accessed."""
        if isinstance(root_value, TypedValue):
            if hasattr(root_value.typ, attr_name) and callable(
                getattr(root_value.typ, attr_name)
            ):
                return
            if self.should_perform_async_checks():
                self._check_attribute_access_in_async(root_value, attr_name, node)

    def _check_attribute_access_in_async(
        self, root_value: Value, attr_name: str, node: ast.Attribute
    ) -> None:
        if isinstance(root_value, TypedValue):
            if not (
                hasattr(root_value.typ, attr_name)
                and isinstance(getattr(root_value.typ, attr_name), property)
            ):
                return
            async_names = ("get_" + attr_name, "is_" + attr_name)
            for async_name in async_names:
                if hasattr(root_value.typ, async_name) and asynq.is_async_fn(
                    getattr(root_value.typ, async_name)
                ):
                    replacement_call = _stringify_async_fn(
                        UnboundMethodValue(async_name, Composite(root_value), "asynq")
                    )
                    method_node = ast.Attribute(value=node.value, attr=async_name)
                    func_node = ast.Attribute(value=method_node, attr="asynq")
                    kwargs = {"args": [], "keywords": []}
                    call_node = ast.Call(func=func_node, **kwargs)
                    replacement_node = ast.Yield(value=call_node)
                    self._show_impure_async_error(
                        node,
                        replacement_call=replacement_call,
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
        if self.current_func_name in self.config.METHODS_NOT_CHECKED_FOR_ASYNQ:
            return False
        return self.should_check_class_for_async(self.current_class)

    def should_check_class_for_async(self, cls: type) -> bool:
        """Returns whether we should perform async checks on all methods on this class."""
        for base_cls in self.config.BASE_CLASSES_CHECKED_FOR_ASYNQ:
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
        return "%s.asynq" % (_stringify_obj(value.val),)
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
        return "%s.%s" % (_stringify_obj(obj.__self__), obj.__name__)
    elif hasattr(obj, "decorator") and hasattr(obj, "instance"):
        if hasattr(obj.instance, "__name__"):
            cls = obj.instance
        else:
            cls = type(obj.instance)
        return "%s.%s" % (_stringify_obj(cls), obj.decorator.fn.__name__)
    elif isinstance(obj, super):
        # self might not always be correct, but it's close enough
        return "super(%s, self)" % _stringify_obj(obj.__self_class__)
    else:
        return f"{obj.__module__}.{obj.__name__}"


def replace_func_on_call_node(node: ast.Call, new_func: ast.expr) -> ast.Call:
    return ast.Call(func=new_func, args=node.args, keywords=node.keywords)
