"""

An object that represents a type.

"""
import inspect
from dataclasses import dataclass, field
from typing import Container, Dict, Sequence, Set, Union
from unittest import mock

from .safe import safe_in, safe_isinstance, safe_issubclass
from .value import (
    BoundsMap,
    CanAssign,
    CanAssignContext,
    CanAssignError,
    KnownValue,
    stringify_object,
    SubclassValue,
    TypedValue,
    unify_bounds_maps,
    UNINITIALIZED_VALUE,
    Value,
)


def get_mro(typ: Union[type, super]) -> Sequence[type]:
    if isinstance(typ, super):
        typ_for_mro = typ.__thisclass__
    else:
        typ_for_mro = typ
    try:
        return inspect.getmro(typ_for_mro)
    except AttributeError:
        # It's not actually a class.
        return []


@dataclass
class TypeObject:
    typ: Union[type, super, str]
    base_classes: Set[Union[type, str]] = field(default_factory=set)
    is_protocol: bool = False
    protocol_members: Set[str] = field(default_factory=set)
    is_thrift_enum: bool = field(init=False)
    is_universally_assignable: bool = field(init=False)
    artificial_bases: Set[type] = field(default_factory=set, init=False)
    _protocol_positive_cache: Dict[Value, BoundsMap] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.typ, str):
            # Synthetic type
            self.is_universally_assignable = False
            self.is_thrift_enum = False
            return
        if isinstance(self.typ, super):
            self.is_universally_assignable = False
        else:
            assert isinstance(self.typ, type), repr(self.typ)
            self.is_universally_assignable = issubclass(self.typ, mock.NonCallableMock)
        self.is_thrift_enum = hasattr(self.typ, "_VALUES_TO_NAMES")
        self.base_classes |= set(get_mro(self.typ))
        # As a special case, the Python type system treats int as
        # a subtype of float, and both int and float as subtypes of complex.
        if self.typ is int:
            self.artificial_bases.add(float)
            self.artificial_bases.add(complex)
        if self.typ is float:
            self.artificial_bases.add(complex)
        if self.is_thrift_enum:
            self.artificial_bases.add(int)
        self.base_classes |= self.artificial_bases

    def is_assignable_to_type(self, typ: type) -> bool:
        for base in self.base_classes:
            if isinstance(base, str):
                continue
            else:
                if safe_issubclass(base, typ):
                    return True
        return self.is_universally_assignable

    def is_assignable_to_type_object(self, other: "TypeObject") -> bool:
        if isinstance(other.typ, super):
            return False
        if isinstance(other.typ, str):
            return (
                self.is_universally_assignable
                # TODO actually check protocols
                or other.is_protocol
                or other.typ in self.base_classes
            )
        return self.is_assignable_to_type(other.typ)

    def can_assign(
        self,
        self_val: Value,
        other_val: Union[KnownValue, TypedValue, SubclassValue],
        ctx: CanAssignContext,
    ) -> CanAssign:
        other = other_val.get_type_object(ctx)
        if other.is_universally_assignable:
            return {}
        if isinstance(self.typ, super):
            if isinstance(other.typ, super):
                return {}
            return CanAssignError(f"Cannot assign to super object {self}")
        if not self.is_protocol:
            if other.is_protocol:
                if self.typ is object:
                    return {}
                return CanAssignError(
                    f"Cannot assign protocol {other_val} to non-protocol {self}"
                )
            if isinstance(self.typ, str):
                if safe_in(self.typ, other.base_classes):
                    return {}
                return CanAssignError(f"Cannot assign {other_val} to {self}")
            else:
                for base in other.base_classes:
                    if base is self.typ:
                        return {}
                    if isinstance(base, type) and safe_issubclass(base, self.typ):
                        return {}
                return CanAssignError(f"Cannot assign {other_val} to {self}")
        else:
            if isinstance(other.typ, super):
                return CanAssignError(
                    f"Cannot assign super object {other_val} to protocol {self}"
                )
            bounds_map = self._protocol_positive_cache.get(other_val)
            if bounds_map is not None:
                return bounds_map
            # This is a guard against infinite recursion if the Protocol is recursive
            if ctx.can_assume_compatibility(self, other):
                return {}
            with ctx.assume_compatibility(self, other):
                result = self._is_compatible_with_protocol(self_val, other_val, ctx)
                if isinstance(result, CanAssignError) and other.artificial_bases:
                    for base in other.artificial_bases:
                        subresult = self._is_compatible_with_protocol(
                            self_val, TypedValue(base), ctx
                        )
                        if not isinstance(subresult, CanAssignError):
                            result = subresult
                            break
            if not isinstance(result, CanAssignError):
                self._protocol_positive_cache[other_val] = result
            return result

    def _is_compatible_with_protocol(
        self, self_val: Value, other_val: Value, ctx: CanAssignContext
    ) -> CanAssign:
        bounds_maps = []
        for member in self.protocol_members:
            expected = ctx.get_attribute_from_value(
                self_val, member, prefer_typeshed=True
            )
            # For __call__, we check compatibility with the other object itself.
            if member == "__call__":
                actual = other_val
            else:
                actual = ctx.get_attribute_from_value(other_val, member)
            if actual is UNINITIALIZED_VALUE:
                can_assign = CanAssignError(f"{other_val} has no attribute {member!r}")
            else:
                can_assign = expected.can_assign(actual, ctx)
                if isinstance(can_assign, CanAssignError):
                    can_assign = CanAssignError(
                        f"Value of protocol member {member!r} conflicts", [can_assign]
                    )

            if isinstance(can_assign, CanAssignError):
                return can_assign
            bounds_maps.append(can_assign)
        return unify_bounds_maps(bounds_maps)

    def is_instance(self, obj: object) -> bool:
        """Whether obj is an instance of this type."""
        return safe_isinstance(obj, self.typ)

    def is_exactly(self, types: Container[type]) -> bool:
        return self.typ in types

    def is_metatype_of(self, other: "TypeObject") -> bool:
        if isinstance(self.typ, type) and isinstance(other.typ, type):
            return issubclass(self.typ, type) and safe_isinstance(other.typ, self.typ)
        else:
            # TODO handle this for synthetic types (if necessary)
            return False

    def has_attribute(self, attr: str, ctx: CanAssignContext) -> bool:
        """Whether this type definitely has this attribute."""
        if self.is_protocol:
            return attr in self.protocol_members
        # We don't use ctx.get_attribute because that may have false positives.
        for base in self.base_classes:
            try:
                present = attr in base.__dict__
            except Exception:
                present = False
            if present:
                return True
        return False

    def __str__(self) -> str:
        base = stringify_object(self.typ)
        if self.is_protocol:
            return (
                f"{base} (Protocol with members"
                f" {', '.join(map(repr, self.protocol_members))})"
            )
        return base
