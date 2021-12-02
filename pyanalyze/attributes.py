"""

Code for retrieving the value of attributes.

"""
import ast
import asynq
from dataclasses import dataclass
from enum import Enum
import inspect
import qcore
import sys
import types
from typing import Any, Generic, Sequence, Tuple, Optional, Union


from .annotations import type_from_runtime, Context
from .safe import safe_isinstance, safe_issubclass
from .signature import Signature, MaybeSignature
from .stacked_scopes import Composite
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    GenericBases,
    HasAttrExtension,
    KnownValueWithTypeVars,
    Value,
    KnownValue,
    GenericValue,
    UNINITIALIZED_VALUE,
    MultiValuedValue,
    UnboundMethodValue,
    SubclassValue,
    TypedValue,
    TypeVarValue,
)

# these don't appear to be in the standard types module
SlotWrapperType = type(type.__init__)
MethodDescriptorType = type(list.append)
NoneType = type(None)


@dataclass
class AttrContext:
    root_composite: Composite
    attr: str

    @property
    def root_value(self) -> Value:
        return self.root_composite.value

    def record_usage(self, obj: Any, val: Value) -> None:
        pass

    def record_attr_read(self, obj: Any) -> None:
        pass

    def should_ignore_class_attribute(self, obj: Any) -> bool:
        return False

    def get_property_type_from_config(self, obj: Any) -> Value:
        return AnyValue(AnySource.inference)

    def get_property_type_from_argspec(self, obj: Any) -> Value:
        return AnyValue(AnySource.inference)

    def get_attribute_from_typeshed(self, typ: type, *, on_class: bool) -> Value:
        return UNINITIALIZED_VALUE

    def get_attribute_from_typeshed_recursively(
        self, fq_name: str, *, on_class: bool
    ) -> Tuple[Value, object]:
        return UNINITIALIZED_VALUE, None

    def should_ignore_none_attributes(self) -> bool:
        return False

    def get_signature(self, obj: object) -> Optional[Signature]:
        return None

    def get_generic_bases(
        self, typ: Union[type, str], generic_args: Sequence[Value]
    ) -> GenericBases:
        return {}


def get_attribute(ctx: AttrContext) -> Value:
    root_value = ctx.root_value
    if isinstance(root_value, TypeVarValue):
        root_value = root_value.get_fallback_value()
    elif isinstance(root_value, AnnotatedValue):
        root_value = root_value.value
    if isinstance(root_value, KnownValue):
        attribute_value = _get_attribute_from_known(root_value.val, ctx)
    elif isinstance(root_value, TypedValue):
        if (
            isinstance(root_value, CallableValue)
            and ctx.attr == "asynq"
            and root_value.signature.is_asynq
        ):
            return root_value.get_asynq_value()
        if isinstance(root_value, GenericValue):
            args = root_value.args
        else:
            args = ()
        if isinstance(root_value.typ, str):
            attribute_value = _get_attribute_from_synthetic_type(
                root_value.typ, args, ctx
            )
        else:
            attribute_value = _get_attribute_from_typed(root_value.typ, args, ctx)
    elif isinstance(root_value, SubclassValue):
        if isinstance(root_value.typ, TypedValue):
            if isinstance(root_value.typ.typ, str):
                # TODO handle synthetic types
                return AnyValue(AnySource.inference)
            attribute_value = _get_attribute_from_subclass(root_value.typ.typ, ctx)
        elif isinstance(root_value.typ, AnyValue):
            attribute_value = AnyValue(AnySource.from_another)
        else:
            attribute_value = _get_attribute_from_known(type, ctx)
    elif isinstance(root_value, UnboundMethodValue):
        attribute_value = _get_attribute_from_unbound(root_value, ctx)
    elif isinstance(root_value, AnyValue):
        attribute_value = AnyValue(AnySource.from_another)
    elif isinstance(root_value, MultiValuedValue):
        raise TypeError("caller should unwrap MultiValuedValue")
    else:
        attribute_value = UNINITIALIZED_VALUE
    if (
        isinstance(attribute_value, AnyValue) or attribute_value is UNINITIALIZED_VALUE
    ) and isinstance(ctx.root_value, AnnotatedValue):
        for guard in ctx.root_value.get_metadata_of_type(HasAttrExtension):
            if guard.attribute_name == KnownValue(ctx.attr):
                return guard.attribute_type
    return attribute_value


def may_have_dynamic_attributes(typ: type) -> bool:
    """These types have typeshed stubs, but instances may have other attributes."""
    if typ is type or typ is super or typ is types.FunctionType:
        return True
    return False


def _get_attribute_from_subclass(typ: type, ctx: AttrContext) -> Value:
    ctx.record_attr_read(typ)

    # First check values that are special in Python
    if ctx.attr == "__class__":
        return KnownValue(type(typ))
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    elif ctx.attr == "__bases__":
        return GenericValue(tuple, [SubclassValue(TypedValue(object))])
    result, _, should_unwrap = _get_attribute_from_mro(typ, ctx, on_class=True)
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
        return AnyValue(AnySource.inference)  # can't figure out what this will return
    elif ctx.should_ignore_class_attribute(cls_val):
        return AnyValue(AnySource.error)
    else:
        return KnownValue(cls_val)


def _get_attribute_from_synthetic_type(
    fq_name: str, generic_args: Sequence[Value], ctx: AttrContext
) -> Value:
    # First check values that are special in Python
    if ctx.attr == "__class__":
        # TODO: a KnownValue for synthetic types?
        return AnyValue(AnySource.inference)
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    result, provider = ctx.get_attribute_from_typeshed_recursively(
        fq_name, on_class=False
    )
    return _substitute_typevars(fq_name, generic_args, result, provider, ctx)


def _get_attribute_from_typed(
    typ: type, generic_args: Sequence[Value], ctx: AttrContext
) -> Value:
    ctx.record_attr_read(typ)

    # First check values that are special in Python
    if ctx.attr == "__class__":
        return KnownValue(typ)
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    result, provider, should_unwrap = _get_attribute_from_mro(typ, ctx, on_class=False)
    result = _substitute_typevars(typ, generic_args, result, provider, ctx)
    if should_unwrap:
        result = _unwrap_value_from_typed(result, typ, ctx)
    ctx.record_usage(typ, result)
    return result


def _substitute_typevars(
    typ: Union[type, str],
    generic_args: Sequence[Value],
    result: Value,
    provider: object,
    ctx: AttrContext,
) -> Value:
    if isinstance(typ, (type, str)):
        generic_bases = ctx.get_generic_bases(typ, generic_args)
    else:
        generic_bases = {}
    if provider in generic_bases:
        result = result.substitute_typevars(generic_bases[provider])
    if generic_args and typ in generic_bases:
        typevars = [
            val.typevar
            for val in generic_bases[typ].values()
            if isinstance(val, TypeVarValue)
        ]
        tv_map = dict(zip(typevars, generic_args))
        result = result.substitute_typevars(tv_map)
    return result


def _unwrap_value_from_typed(result: Value, typ: type, ctx: AttrContext) -> Value:
    if not isinstance(result, KnownValue):
        return result
    typevars = result.typevars if isinstance(result, KnownValueWithTypeVars) else None
    cls_val = result.val
    if isinstance(cls_val, property):
        typ = ctx.get_property_type_from_config(cls_val)
        if not isinstance(typ, AnyValue):
            return typ
        return ctx.get_property_type_from_argspec(cls_val)
    elif qcore.inspection.is_classmethod(cls_val):
        return result
    elif inspect.ismethod(cls_val):
        return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif inspect.isfunction(cls_val):
        # either a staticmethod or an unbound method
        try:
            descriptor = inspect.getattr_static(typ, ctx.attr)
        except AttributeError:
            # probably a super call; assume unbound method
            if ctx.attr != "__new__":
                return UnboundMethodValue(
                    ctx.attr, ctx.root_composite, typevars=typevars
                )
            else:
                # __new__ is implicitly a staticmethod
                return result
        if isinstance(descriptor, staticmethod) or ctx.attr == "__new__":
            return result
        else:
            return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif isinstance(cls_val, (MethodDescriptorType, SlotWrapperType)):
        # built-in method; e.g. scope_lib.tests.SimpleDatabox.get
        return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif (
        _static_hasattr(cls_val, "decorator")
        and _static_hasattr(cls_val, "instance")
        and not isinstance(cls_val.instance, type)
    ):
        # non-static method
        return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif asynq.is_async_fn(cls_val):
        # static or class method
        return result
    elif _static_hasattr(cls_val, "__get__"):
        typeshed_type = ctx.get_attribute_from_typeshed(typ, on_class=False)
        if typeshed_type is not UNINITIALIZED_VALUE:
            return typeshed_type
        return ctx.get_property_type_from_config(cls_val)
    elif ctx.should_ignore_class_attribute(cls_val):
        return AnyValue(AnySource.error)
    else:
        return result


def _get_attribute_from_known(obj: object, ctx: AttrContext) -> Value:
    ctx.record_attr_read(type(obj))

    if (obj is None or obj is NoneType) and ctx.should_ignore_none_attributes():
        # This usually indicates some context is set to None
        # in the module and initialized later.
        return AnyValue(AnySource.error)

    # Type alias to Any
    if obj is Any:
        return AnyValue(AnySource.explicit)

    # Avoid generating huge Union type with the actual value
    if obj is sys and ctx.attr == "modules":
        return GenericValue(dict, [TypedValue(str), TypedValue(types.ModuleType)])

    result, _, _ = _get_attribute_from_mro(obj, ctx, on_class=True)
    if isinstance(obj, (types.ModuleType, type)):
        ctx.record_usage(obj, result)
    else:
        ctx.record_usage(type(obj), result)
    return result


def _get_attribute_from_unbound(
    root_value: UnboundMethodValue, ctx: AttrContext
) -> Value:
    method = root_value.get_method()
    if method is None:
        return AnyValue(AnySource.inference)
    try:
        getattr(method, ctx.attr)
    except AttributeError:
        return UNINITIALIZED_VALUE
    result = UnboundMethodValue(
        root_value.attr_name, root_value.composite, secondary_attr_name=ctx.attr
    )
    ctx.record_usage(type(method), result)
    return result


@dataclass
class AnnotationsContext(Context):
    attr_ctx: AttrContext
    cls: object

    def __post_init__(self) -> None:
        super().__init__()

    def get_name(self, node: ast.Name) -> Value:
        try:
            if isinstance(self.cls, types.ModuleType):
                globals = self.cls.__dict__
            else:
                globals = sys.modules[self.cls.__module__].__dict__
        except Exception:
            return AnyValue(AnySource.error)
        else:
            return self.get_name_from_globals(node.id, globals)

    def get_signature(self, callable: object) -> MaybeSignature:
        return self.attr_ctx.get_signature(callable)


def _get_attribute_from_mro(
    typ: object, ctx: AttrContext, on_class: bool
) -> Tuple[Value, object, bool]:
    # Then go through the MRO and find base classes that may define the attribute.
    if safe_isinstance(typ, type) and safe_issubclass(typ, Enum):
        # Special case, to avoid picking an attribute of Enum instances (e.g., name)
        # over an Enum member. Ideally we'd have a more principled way to support this
        # but I haven't thought of one.
        try:
            return KnownValue(getattr(typ, ctx.attr)), typ, True
        except Exception:
            pass
    elif safe_isinstance(typ, types.ModuleType):
        try:
            annotation = typ.__annotations__[ctx.attr]
        except Exception:
            # Module doesn't have annotations or it's not in there
            pass
        else:
            attr_type = type_from_runtime(annotation, ctx=AnnotationsContext(ctx, typ))
            if attr_type != AnyValue(AnySource.incomplete_annotation):
                return (attr_type, typ, False)

    try:
        mro = list(typ.mro())
    except Exception:
        # broken mro method
        pass
    else:
        for base_cls in mro:
            # On 3.6 (before PEP 560), the MRO for classes inheriting from typing generics
            # includes a bunch of classes in the typing module that
            # don't have any attributes we care about.
            if (
                sys.version_info < (3, 7)
                and isinstance(base_cls, type)
                and base_cls.__module__ == "typing"
                and Generic in base_cls.mro()
            ):
                continue
            try:
                # Make sure to use only __annotations__ that are actually on this
                # class, not ones inherited from a base class.
                annotation = base_cls.__dict__["__annotations__"][ctx.attr]
            except Exception:
                # no __annotations__, or it's not a dict, or the attr isn't there
                pass
            else:
                attr_type = type_from_runtime(
                    annotation, ctx=AnnotationsContext(ctx, base_cls)
                )
                if attr_type != AnyValue(AnySource.incomplete_annotation):
                    return (attr_type, base_cls, False)
            try:
                # Make sure we use only the object from this class, but do invoke
                # the descriptor protocol with getattr.
                base_cls.__dict__[ctx.attr]
                return KnownValue(getattr(typ, ctx.attr)), base_cls, True
            except Exception:
                pass

            typeshed_type = ctx.get_attribute_from_typeshed(base_cls, on_class=on_class)
            if typeshed_type is not UNINITIALIZED_VALUE:
                return typeshed_type, base_cls, False

    attrs_type = get_attrs_attribute(typ, ctx)
    if attrs_type is not None:
        return attrs_type, typ, False

    # Even if we didn't find it any __dict__, maybe getattr() finds it directly.
    try:
        return KnownValue(getattr(typ, ctx.attr)), typ, True
    except Exception:
        pass

    return UNINITIALIZED_VALUE, object, False


def _static_hasattr(value: object, attr: str) -> bool:
    """Returns whether this value has the given attribute, ignoring __getattr__ overrides."""
    try:
        object.__getattribute__(value, attr)
    except AttributeError:
        return False
    else:
        return True


def get_attrs_attribute(typ: object, ctx: AttrContext) -> Optional[Value]:
    try:
        if hasattr(typ, "__attrs_attrs__"):
            for attr_attr in typ.__attrs_attrs__:
                if attr_attr.name == ctx.attr:
                    if attr_attr.type is not None:
                        return type_from_runtime(
                            attr_attr.type, ctx=AnnotationsContext(ctx, typ)
                        )
                    else:
                        return AnyValue(AnySource.unannotated)
    except Exception:
        # Guard against silly objects throwing exceptions on hasattr()
        # or similar shenanigans.
        pass
    return None
