"""

Structured configuration options.

"""
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, Dict, Mapping, Sequence, Generic, Tuple, Type, TypeVar


from .config import Config
from .error_code import ErrorCode, DISABLED_BY_DEFAULT, ERROR_DESCRIPTION

T = TypeVar("T")
ModulePath = Tuple[str, ...]


class InvalidConfigOption(Exception):
    """Raised when an invalid config option is encountered."""

    @classmethod
    def from_parser(
        cls, option_cls: Type["ConfigOption"], expected: str, value: object
    ) -> "InvalidConfigOption":
        return cls(
            f"Invalid value for option {option_cls.name}: expected {expected} but got"
            f" {value!r}"
        )


class NotFound(Exception):
    """Raised if no value is found for an option."""


@dataclass
class ConfigOption(Generic[T]):
    registry: ClassVar[Dict[str, Type["ConfigOption"]]] = {}

    name: ClassVar[str]
    is_global: ClassVar[bool] = False
    default_value: ClassVar[T]
    value: T
    applicable_to: ModulePath = ()
    from_command_line: bool = False

    def __init_subclass__(cls) -> None:
        if hasattr(cls, "name"):
            if cls.name in cls.registry:
                raise ValueError(f"Duplicate option {cls.name}")
            cls.registry[cls.name] = cls

    @classmethod
    def parse(cls: "Type[ConfigOption[T]]", data: object) -> T:
        raise NotImplementedError

    @classmethod
    def get_value_from_instances(
        cls: "Type[ConfigOption[T]]",
        instances: Sequence["ConfigOption[T]"],
        module_path: ModulePath,
    ) -> T:
        for instance in instances:
            if instance.is_applicable_to(module_path):
                return instance.value
        raise NotFound

    def is_applicable_to(self, module_path: ModulePath) -> bool:
        return module_path[: len(self.applicable_to)] == self.applicable_to

    def sort_key(self) -> Tuple[object, ...]:
        """We sort with the most specific option first."""
        return (
            not self.from_command_line,  # command line options first
            -len(self.applicable_to),  # longest options first
        )


class BooleanOption(ConfigOption[bool]):
    default_value = False

    @classmethod
    def parse(cls: "Type[BooleanOption]", data: object) -> bool:
        if isinstance(data, bool):
            return data
        raise InvalidConfigOption.from_parser(cls, "bool", data)


class StringSequenceOption(ConfigOption[Sequence[str]]):
    default_value = ()

    @classmethod
    def parse(cls: "Type[StringSequenceOption]", data: object) -> Sequence[str]:
        if isinstance(data, (list, tuple)) and all(
            isinstance(elt, str) for elt in data
        ):
            return data
        raise InvalidConfigOption.from_parser(cls, "sequence of strings", data)


class Paths(StringSequenceOption):
    """Paths that pyanalyze should type check."""

    name = "paths"
    is_global = True


class ImportPaths(StringSequenceOption):
    """Directories that pyanalyze may import from."""

    name = "import_paths"
    is_global = True


class EnforceNoUnused(BooleanOption):
    """If true, an error is raised when pyanalyze finds any unused objects."""

    name = "enforce_no_unused"
    is_global = True


for _code in ErrorCode:
    type(
        _code.name,
        (BooleanOption,),
        {
            "__doc__": ERROR_DESCRIPTION[_code],
            "name": _code.name,
            "default_value": _code not in DISABLED_BY_DEFAULT,
        },
    )


@dataclass
class Options:
    options: Mapping[str, Sequence[ConfigOption]]
    fallback: Config
    module_path: ModulePath = ()

    @classmethod
    def from_option_list(
        cls, instances: Sequence[ConfigOption], fallback: Config
    ) -> "Options":
        by_name = defaultdict(list)
        for instance in instances:
            by_name[instance.name].append(instance)
        options = {
            name: sorted(instances, key=lambda i: i.sort_key())
            for name, instances in by_name.items()
        }
        return Options(options, fallback)

    def for_module(self, module_path: ModulePath) -> "Options":
        return Options(self.options, self.fallback, module_path)

    def get_value_for(self, option: Type[ConfigOption[T]]) -> T:
        instances = self.options.get(option.name, ())
        try:
            return option.get_value_from_instances(instances, self.module_path)
        except NotFound:
            return option.default_value

    def _get_value_for_no_default(self, option: Type[ConfigOption[T]]) -> T:
        instances = self.options.get(option.name, ())
        return option.get_value_from_instances(instances, self.module_path)

    def is_error_code_enabled(self, code: ErrorCode) -> bool:
        option = ConfigOption.registry[code.name]
        try:
            return self._get_value_for_no_default(option)
        except NotFound:
            if code in self.fallback.ENABLED_ERRORS:
                return True
            if code in self.fallback.DISABLED_ERRORS:
                return False
            return option.default_value
