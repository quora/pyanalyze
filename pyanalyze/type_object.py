"""

An object that represents a type.

"""
from dataclasses import dataclass, field
import inspect
from typing import Container, Set, Sequence, Union
from unittest import mock

from .safe import safe_isinstance, safe_issubclass, safe_in
from .value import (
    UNINITIALIZED_VALUE,
    CanAssign,
    CanAssignContext,
    CanAssignError,
    TypedValue,
    stringify_object,
    unify_typevar_maps,
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
        if self.typ is int:
            # As a special case, the Python type system treats int as
            # a subtype of float.
            self.base_classes.add(float)

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

    def can_assign_type_object(
        self, other: "TypeObject", ctx: CanAssignContext
    ) -> CanAssign:
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
                    f"Cannot assign protocol {other} to non-protocol {self}"
                )
            if isinstance(self.typ, str):
                if safe_in(self.typ, other.base_classes):
                    return {}
                return CanAssignError(f"Cannot assign {other} to {self}")
            else:
                for base in other.base_classes:
                    if base is self.typ:
                        return {}
                    if isinstance(base, type) and safe_issubclass(base, self.typ):
                        return {}
                return CanAssignError(f"Cannot assign {other} to {self}")
        else:
            self_val = TypedValue(self.typ)
            if isinstance(other.typ, super):
                return CanAssignError(
                    f"Cannot assign super object {other} to protocol {self}"
                )
            other_val = TypedValue(other.typ)
            tv_maps = []
            for member in self.protocol_members:
                expected = ctx.get_attribute_from_value(self_val, member)
                actual = ctx.get_attribute_from_value(other_val, member)
                if actual is UNINITIALIZED_VALUE:
                    return CanAssignError(f"{other} has no attribute {member!r}")
                tv_map = expected.can_assign(actual, ctx)
                if isinstance(tv_map, CanAssignError):
                    return CanAssignError(
                        f"Value of protocol member {member!r} conflicts", [tv_map]
                    )
                tv_maps.append(tv_map)
            return unify_typevar_maps(tv_maps)

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

    def __str__(self) -> str:
        if isinstance(self.typ, str):
            return self.typ
        else:
            return stringify_object(self.typ)
