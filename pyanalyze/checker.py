from dataclasses import dataclass, field
from typing import Dict, Union


"""

The checker maintains global state that is preserved across different modules.

"""
from .config import Config
from .type_object import TypeObject


from dataclasses import dataclass, field
from typing import Set


@dataclass
class Checker:
    config: Config
    type_object_cache: Dict[Union[type, super], TypeObject] = field(
        default_factory=dict, init=False, repr=False
    )

    def get_additional_bases(self, typ: Union[type, super]) -> Set[type]:
        return self.config.get_additional_bases(typ)

    def make_type_object(self, typ: Union[type, super]) -> TypeObject:
        try:
            in_cache = typ in self.type_object_cache
        except Exception:
            return TypeObject(typ, self.get_additional_bases(typ))
        if in_cache:
            return self.type_object_cache[typ]
        type_object = TypeObject(typ, self.get_additional_bases(typ))
        self.type_object_cache[typ] = type_object
        return type_object
