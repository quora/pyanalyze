"""

The checker maintains global state that is preserved across different modules.

"""
import itertools
from contextlib import contextmanager
from dataclasses import dataclass, field
import sys
from typing import Iterable, Iterator, List, Set, Tuple, Union, Dict

from .value import TypedValue
from .arg_spec import ArgSpecCache
from .config import Config
from .reexport import ImplicitReexportTracker
from .safe import is_instance_of_typing_name, is_typing_name, safe_getattr
from .type_object import TypeObject, get_mro


@dataclass
class Checker:
    config: Config
    arg_spec_cache: ArgSpecCache = field(init=False)
    reexport_tracker: ImplicitReexportTracker = field(init=False)
    type_object_cache: Dict[Union[type, super, str], TypeObject] = field(
        default_factory=dict, init=False, repr=False
    )
    assumed_compatibilities: List[Tuple[TypeObject, TypeObject]] = field(
        default_factory=list
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
            # Is it marked as a protocol in stubs? If so, use the stub definition.
            if self.arg_spec_cache.ts_finder.is_protocol(typ):
                bases = self._get_typeshed_bases(typ)
                return TypeObject(
                    typ,
                    additional_bases,
                    is_protocol=True,
                    protocol_members=self._get_protocol_members(bases),
                )
            # Is it a protocol at runtime?
            if is_instance_of_typing_name(typ, "_ProtocolMeta") and safe_getattr(
                typ, "_is_protocol", False
            ):
                bases = get_mro(typ)
                members = set(
                    itertools.chain.from_iterable(
                        _extract_protocol_members(base) for base in bases
                    )
                )
                return TypeObject(
                    typ, additional_bases, is_protocol=True, protocol_members=members
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

    def can_assume_compatibility(self, left: TypeObject, right: TypeObject) -> bool:
        return (left, right) in self.assumed_compatibilities

    @contextmanager
    def assume_compatibility(
        self, left: TypeObject, right: TypeObject
    ) -> Iterator[None]:
        """Context manager that notes that left and right can be assumed to be compatible."""
        pair = (left, right)
        self.assumed_compatibilities.append(pair)
        try:
            yield
        finally:
            new_pair = self.assumed_compatibilities.pop()
            assert pair == new_pair


EXCLUDED_PROTOCOL_MEMBERS = {
    "__abstractmethods__",
    "__annotations__",
    "__dict__",
    "__doc__",
    "__init__",
    "__new__",
    "__module__",
    "__parameters__",
    "__subclasshook__",
    "__weakref__",
    "_abc_impl",
    "_abc_cache",
    "_is_protocol",
    "__next_in_mro__",
    "_abc_generic_negative_cache_version",
    "__orig_bases__",
    "__args__",
    "_abc_registry",
    "__extra__",
    "_abc_generic_negative_cache",
    "__origin__",
    "__tree_hash__",
    "_gorg",
    "_is_runtime_protocol",
}


def _extract_protocol_members(typ: type) -> Set[str]:
    if (
        typ is object
        or is_typing_name(typ, "Generic")
        or is_typing_name(typ, "Protocol")
    ):
        return set()
    members = set(typ.__dict__) - EXCLUDED_PROTOCOL_MEMBERS
    # Starting in 3.10 __annotations__ always exists on types
    if sys.version_info >= (3, 10) or hasattr(typ, "__annotations__"):
        members |= set(typ.__annotations__)
    return members
