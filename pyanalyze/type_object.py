"""

An object that represents a type.

"""
from dataclasses import dataclass, field
import inspect
from typing import Set, Dict, Sequence, Union
from unittest import mock

from .safe import safe_issubclass

_cache: Dict[type, "TypeObject"] = {}


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
    typ: Union[type, super]
    base_classes: Set[type] = field(default_factory=set)
    is_thrift_enum: bool = field(init=False)
    is_universally_assignable: bool = field(init=False)

    def __post_init__(self) -> None:
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

    @classmethod
    def make(cls, typ: type) -> "TypeObject":
        try:
            in_cache = typ in _cache
        except Exception:
            return cls(typ)
        if in_cache:
            return _cache[typ]
        type_object = cls(typ)
        _cache[typ] = type_object
        return type_object

    def is_assignable_to_type(self, typ: type) -> bool:
        for base in self.base_classes:
            if safe_issubclass(base, typ):
                return True
        return self.is_universally_assignable
