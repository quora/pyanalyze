"""

The checker maintains global state that is preserved across different modules.

"""
from .value import TypedValue
from .arg_spec import ArgSpecCache
from .config import Config
from .reexport import ImplicitReexportTracker
from .type_object import TypeObject

from dataclasses import dataclass, field
from typing import Set, Union, Dict


@dataclass
class Checker:
    config: Config
    arg_spec_cache: ArgSpecCache = field(init=False)
    reexport_tracker: ImplicitReexportTracker = field(init=False)
    type_object_cache: Dict[Union[type, super, str], TypeObject] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.arg_spec_cache = ArgSpecCache(self.config)
        self.reexport_tracker = ImplicitReexportTracker(self.config)

    def get_additional_bases(self, typ: Union[type, super]) -> Set[type]:
        return self.config.get_additional_bases(typ)

    def make_type_object(self, typ: Union[type, super, str]) -> TypeObject:
        try:
            in_cache = typ in self.type_object_cache
        except Exception:
            return self._build_type_object(typ)
        if in_cache:
            return self.type_object_cache[typ]
        type_object = self._build_type_object(typ)
        self.type_object_cache[typ] = type_object
        return type_object

    def _build_type_object(self, typ: Union[type, super, str]) -> TypeObject:
        if isinstance(typ, str):
            # Synthetic type
            bases = self.arg_spec_cache.ts_finder.get_bases_recursively(typ)
            return TypeObject(
                typ, set(base.typ for base in bases if isinstance(base, TypedValue))
            )
        return TypeObject(typ, self.get_additional_bases(typ))
