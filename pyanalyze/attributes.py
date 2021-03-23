"""

Code for retrieving the value of attributes.

"""
import ast
import asynq
from dataclasses import dataclass
import inspect
import qcore
import types
from typing import Any, Tuple

from .annotations import type_from_runtime
from .value import (
    Value,
    KnownValue,
    GenericValue,
    UNINITIALIZED_VALUE,
    UNRESOLVED_VALUE,
    MultiValuedValue,
    UnboundMethodValue,
    SubclassValue,
    TypedValue,
    TypeVarValue,
    VariableNameValue,
)

# these don't appear to be in the standard types module
SlotWrapperType = type(type.__init__)
MethodDescriptorType = type(list.append)


@dataclass
class AttrContext:
    root_value: Value
    attr: str
    node: ast.AST

    def record_usage(self, obj: Any, val: Value) -> None:
        pass

    def record_attr_read(self, obj: Any) -> None:
        pass

    def should_ignore_class_attribute(self, obj: Any) -> bool:
        return False

    def get_property_type_from_config(self, obj: Any) -> Value:
        return UNRESOLVED_VALUE

    def get_property_type_from_argspec(self, obj: Any) -> Value:
        return UNRESOLVED_VALUE


def get_attribute(ctx: AttrContext) -> Value:
    root_value = ctx.root_value
    if isinstance(root_value, TypeVarValue):
        root_value = root_value.get_fallback_value()
    if isinstance(root_value, KnownValue):
        return _get_attribute_from_known(root_value.val, ctx)
    elif isinstance(root_value, TypedValue):
        return _get_attribute_from_typed(root_value.typ, ctx)
    elif isinstance(root_value, SubclassValue):
        return _get_attribute_from_subclass(root_value.typ, ctx)
    elif isinstance(root_value, UnboundMethodValue):
        return _get_attribute_from_unbound(root_value, ctx)
    elif root_value is UNRESOLVED_VALUE or isinstance(root_value, VariableNameValue):
        return UNRESOLVED_VALUE
    elif isinstance(root_value, MultiValuedValue):
        raise TypeError("caller should unwrap MultiValuedValue")
    else:
        return UNINITIALIZED_VALUE


def _get_attribute_from_subclass(
    typ: type,
    ctx: AttrContext,
) -> Value:
    ctx.record_attr_read(typ)

    # First check values that are special in Python
    if ctx.attr == "__class__":
        return KnownValue(type(typ))
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    elif ctx.attr == "__bases__":
        return GenericValue(tuple, [SubclassValue(object)])
    result, should_unwrap = _get_attribute_from_mro(typ, ctx)
    if should_unwrap:
        result = _unwrap_value_from_subclass(result, ctx)
    ctx.record_usage(typ, result)
    return result


def _unwrap_value_from_subclass(result: Value, ctx: AttrContext) -> Value:
    if not isinstance(result, KnownValue):
        return result
    cls_val = result.val
    if (
        qcore.inspection.is_classmethod(cls_val)
        or inspect.ismethod(cls_val)
        or inspect.isfunction(cls_val)
        or isinstance(cls_val, (MethodDescriptorType, SlotWrapperType))
        or (
            # non-static method
            _static_hasattr(cls_val, "decorator")
            and _static_hasattr(cls_val, "instance")
            and not isinstance(cls_val.instance, type)
        )
        or asynq.is_async_fn(cls_val)
    ):
        # static or class method
        return KnownValue(cls_val)
    elif _static_hasattr(cls_val, "__get__"):
        return UNRESOLVED_VALUE  # can't figure out what this will return
    elif ctx.should_ignore_class_attribute(cls_val):
        return UNRESOLVED_VALUE
    else:
        return KnownValue(cls_val)


def _get_attribute_from_typed(typ: type, ctx: AttrContext) -> Value:
    ctx.record_attr_read(typ)

    # First check values that are special in Python
    if ctx.attr == "__class__":
        return KnownValue(typ)
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    result, should_unwrap = _get_attribute_from_mro(typ, ctx)
    if should_unwrap:
        result = _unwrap_value_from_typed(result, typ, ctx)
    ctx.record_usage(typ, result)
    return result


def _unwrap_value_from_typed(result: Value, typ: type, ctx: AttrContext) -> Value:
    if not isinstance(result, KnownValue):
        return result
    cls_val = result.val
    if isinstance(cls_val, property):
        typ = ctx.get_property_type_from_config(cls_val)
        if typ is not UNRESOLVED_VALUE:
            return typ
        return ctx.get_property_type_from_argspec(cls_val)
    elif qcore.inspection.is_classmethod(cls_val):
        return KnownValue(cls_val)
    elif inspect.ismethod(cls_val):
        return UnboundMethodValue(ctx.attr, ctx.root_value)
    elif inspect.isfunction(cls_val):
        # either a staticmethod or an unbound method
        try:
            descriptor = inspect.getattr_static(typ, ctx.attr)
        except AttributeError:
            # probably a super call; assume unbound method
            if ctx.attr != "__new__":
                return UnboundMethodValue(ctx.attr, ctx.root_value)
            else:
                # __new__ is implicitly a staticmethod
                return KnownValue(cls_val)
        if isinstance(descriptor, staticmethod) or ctx.attr == "__new__":
            return KnownValue(cls_val)
        else:
            return UnboundMethodValue(ctx.attr, ctx.root_value)
    elif isinstance(cls_val, (MethodDescriptorType, SlotWrapperType)):
        # built-in method; e.g. scope_lib.tests.SimpleDatabox.get
        return UnboundMethodValue(ctx.attr, ctx.root_value)
    elif (
        _static_hasattr(cls_val, "decorator")
        and _static_hasattr(cls_val, "instance")
        and not isinstance(cls_val.instance, type)
    ):
        # non-static method
        return UnboundMethodValue(ctx.attr, ctx.root_value)
    elif asynq.is_async_fn(cls_val):
        # static or class method
        return KnownValue(cls_val)
    elif _static_hasattr(cls_val, "__get__"):
        return ctx.get_property_type_from_config(cls_val)
    elif ctx.should_ignore_class_attribute(cls_val):
        return UNRESOLVED_VALUE
    else:
        return KnownValue(cls_val)


def _get_attribute_from_known(obj: Any, ctx: AttrContext) -> Value:
    ctx.record_attr_read(type(obj))

    if obj is None:
        # This usually indicates some context is set to None
        # in the module and initialized later.
        return UNRESOLVED_VALUE

    result, _ = _get_attribute_from_mro(obj, ctx)
    if isinstance(obj, (types.ModuleType, type)):
        ctx.record_usage(obj, result)
    else:
        ctx.record_usage(type(obj), result)
    return result


def _get_attribute_from_unbound(root_value: UnboundMethodValue, ctx: AttrContext):
    method = root_value.get_method()
    if method is None:
        return UNRESOLVED_VALUE
    try:
        getattr(method, ctx.attr)
    except AttributeError:
        return UNINITIALIZED_VALUE
    result = UnboundMethodValue(
        root_value.attr_name, root_value.typ, secondary_attr_name=ctx.attr
    )
    ctx.record_usage(type(method), result)
    return result


def _get_attribute_from_mro(typ: type, ctx: AttrContext) -> Tuple[Value, bool]:
    # Then go through the MRO and find base classes that may define the attribute.
    try:
        mro = list(typ.mro())
    except Exception:
        # broken mro method
        pass
    else:
        for base_cls in mro:
            try:
                # Make sure to use only __annotations__ that are actually on this
                # class, not ones inherited from a base class.
                annotation = base_cls.__dict__["__annotations__"][ctx.attr]
            except Exception:
                # no __annotations__, or it's not a dict, or the attr isn't there
                try:
                    # Make sure we use only the object from this class, but do invoke
                    # the descriptor protocol with getattr.
                    base_cls.__dict__[ctx.attr]
                    return KnownValue(getattr(typ, ctx.attr)), True
                except Exception:
                    pass
            else:
                return type_from_runtime(annotation), False

    attrs_type = get_attrs_attribute(typ, ctx.attr)
    if attrs_type is not None:
        return attrs_type, False

    # Even if we didn't find it any __dict__, maybe getattr() finds it directly.
    try:
        return KnownValue(getattr(typ, ctx.attr)), True
    except Exception:
        pass

    return UNINITIALIZED_VALUE, False


def _static_hasattr(value, attr):
    """Returns whether this value has the given attribute, ignoring __getattr__ overrides."""
    try:
        object.__getattribute__(value, attr)
    except AttributeError:
        return False
    else:
        return True


def get_attrs_attribute(typ, attr):
    try:
        if hasattr(typ, "__attrs_attrs__"):
            for attr_attr in typ.__attrs_attrs__:
                if attr_attr.name == attr:
                    if attr_attr.type is not None:
                        return type_from_runtime(attr_attr.type)
                    else:
                        return UNRESOLVED_VALUE
    except Exception:
        # Guard against silly objects throwing exceptions on hasattr()
        # or similar shenanigans.
        pass
    return None
