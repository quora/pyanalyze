from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

"""

Implementation of test_scope's asynq checks.

"""

import ast
import asynq
import contextlib
import enum
import qcore
import inspect
import six
import types

from .error_code import ErrorCode
from .value import KnownValue, TypedValue, UnboundMethodValue


class AsyncFunctionKind(enum.Enum):
    non_async = 0
    normal = 1
    async_proxy = 2
    pure = 3


class AsynqChecker(object):
    def __init__(self, config, module, lines, on_error, log, replace_node):
        self.config = config
        self.module = module
        self.lines = lines
        self.on_error = on_error
        self.log = log
        self.replace_node = replace_node
        self.current_func_name = None
        self.current_class = None
        self.current_async_kind = AsyncFunctionKind.non_async
        self.is_classmethod = False

    # Functions called from test_scope itself

    @contextlib.contextmanager
    def set_func_name(
        self, name, async_kind=AsyncFunctionKind.non_async, is_classmethod=False
    ):
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

    def check_call(self, value, node):
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
            and hasattr(value.typ, value.attr_name + "_async")
        ):
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
                % (_stringify_obj(value.typ), value.attr_name),
                replacement_node=replacement_node,
            )

    def record_attribute_access(self, root_value, attr_name, node):
        """Records that the given attribute of root_value was accessed."""
        if isinstance(root_value, TypedValue):
            if hasattr(root_value.typ, attr_name) and callable(
                getattr(root_value.typ, attr_name)
            ):
                return
            if self.should_perform_async_checks():
                self._check_attribute_access_in_async(root_value, attr_name, node)

    def _check_attribute_access_in_async(self, root_value, attr_name, node):
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
                        UnboundMethodValue(async_name, root_value.typ, "asynq")
                    )
                    method_node = ast.Attribute(value=node.value, attr=async_name)
                    func_node = ast.Attribute(value=method_node, attr="asynq")
                    kwargs = {"args": [], "keywords": []}
                    if six.PY2:
                        kwargs["starargs"] = kwargs["kwargs"] = None
                    call_node = ast.Call(func=func_node, **kwargs)
                    replacement_node = ast.Yield(value=call_node)
                    self._show_impure_async_error(
                        node,
                        replacement_call=replacement_call,
                        replacement_node=replacement_node,
                    )

    def should_perform_async_checks(self):
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

    def should_check_class_for_async(self, cls):
        """Returns whether we should perform async checks on all methods on this class."""
        for base_cls in self.config.BASE_CLASSES_CHECKED_FOR_ASYNQ:
            try:
                if issubclass(cls, base_cls):
                    return True
            except TypeError:
                pass  # old-style class or something
        return False

    def _show_impure_async_error(
        self, node, replacement_call=None, replacement_node=None
    ):
        if replacement_call is None:
            message = "impure async call (you should yield it)"
        else:
            message = "impure async call (you should yield %s)" % replacement_call
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


def is_impure_async_fn(value):
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


def get_pure_async_equivalent(value):
    """Returns the pure-async equivalent of an async function."""
    assert is_impure_async_fn(value), "%s is not an impure async function" % value
    if isinstance(value, KnownValue):
        return "%s.asynq" % (_stringify_obj(value.val),)
    elif isinstance(value, UnboundMethodValue):
        return _stringify_async_fn(
            UnboundMethodValue(value.attr_name, value.typ, "asynq")
        )
    else:
        assert False, "cannot get pure async equivalent of %s" % value


def _stringify_async_fn(value):
    if isinstance(value, KnownValue):
        return _stringify_obj(value.val)
    elif isinstance(value, UnboundMethodValue):
        ret = "%s.%s" % (_stringify_obj(value.typ), value.attr_name)
        if value.secondary_attr_name is not None:
            ret += ".%s" % value.secondary_attr_name
        return ret
    else:
        return str(value)


def _stringify_obj(obj):
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
        return "%s.%s" % (obj.__module__, obj.__name__)


def is_async_classmethod(obj):
    try:
        decorator = obj.decorator
    except AttributeError:
        return False
    else:
        return getattr(decorator, "type", None) is classmethod


def replace_func_on_call_node(node, new_func):
    if six.PY2:
        return ast.Call(
            func=new_func,
            args=node.args,
            keywords=node.keywords,
            starargs=node.starargs,
            kwargs=node.kwargs,
        )
    else:
        return ast.Call(func=new_func, args=node.args, keywords=node.keywords)
