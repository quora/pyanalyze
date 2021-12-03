"""

The checker maintains global state that is preserved across different modules.

"""
import itertools
from .value import TypedValue
from .arg_spec import ArgSpecCache
from .config import Config
from .reexport import ImplicitReexportTracker
from .safe import is_typing_name
from .type_object import TypeObject

from dataclasses import dataclass, field
from typing import Iterable, Set, Union, Dict


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
            bases = self._get_typeshed_bases(typ)
            is_protocol = any(is_typing_name(base, "Protocol") for base in bases)
            if is_protocol:
                protocol_members = self._get_protocol_members(bases)
            else:
                protocol_members = set()
            return TypeObject(
                typ, bases, is_protocol=is_protocol, protocol_members=protocol_members
            )
        elif isinstance(typ, super):
            return TypeObject(typ, self.get_additional_bases(typ))
        else:
            additional_bases = self.get_additional_bases(typ)
            is_protocol = self.arg_spec_cache.ts_finder.is_protocol(typ)
            if is_protocol:
                bases = self._get_typeshed_bases(typ)
                return TypeObject(
                    typ,
                    additional_bases,
                    is_protocol=True,
                    protocol_members=self._get_protocol_members(bases),
                )
            return TypeObject(typ, additional_bases)

    def _get_typeshed_bases(self, typ: Union[type, str]) -> Set[Union[type, str]]:
        base_values = self.arg_spec_cache.ts_finder.get_bases_recursively(typ)
        return set(base.typ for base in base_values if isinstance(base, TypedValue))

    def _get_protocol_members(self, bases: Iterable[Union[type, str]]) -> Set[str]:
        return set(
            itertools.chain.from_iterable(
                self.arg_spec_cache.ts_finder.get_all_attributes(base) for base in bases
            )
        )
