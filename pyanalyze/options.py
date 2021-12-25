"""

Structured configuration options.

"""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Generic,
    Tuple,
    Type,
    TypeVar,
)
import tomli


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
    def parse(cls: "Type[ConfigOption[T]]", data: object, source_path: Path) -> T:
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
    def parse(cls: "Type[BooleanOption]", data: object, source_path: Path) -> bool:
        if isinstance(data, bool):
            return data
        raise InvalidConfigOption.from_parser(cls, "bool", data)


class PathSequenceOption(ConfigOption[Sequence[Path]]):
    default_value = ()

    @classmethod
    def parse(
        cls: "Type[PathSequenceOption]", data: object, source_path: Path
    ) -> Sequence[str]:
        if isinstance(data, (list, tuple)) and all(
            isinstance(elt, str) for elt in data
        ):
            return [(source_path.parent / elt).resolve() for elt in data]
        raise InvalidConfigOption.from_parser(cls, "sequence of strings", data)


class Paths(PathSequenceOption):
    """Paths that pyanalyze should type check."""

    name = "paths"
    is_global = True


class ImportPaths(PathSequenceOption):
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
        cls,
        instances: Sequence[ConfigOption],
        fallback: Config,
        config_file_path: Optional[Path] = None,
    ) -> "Options":
        if config_file_path:
            instances = [*instances, *parse_config_file(config_file_path)]
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

    def is_error_code_enabled_anywhere(self, code: ErrorCode) -> bool:
        option = ConfigOption.registry[code.name]
        instances = self.options.get(option.name, ())
        if any(instance.value for instance in instances):
            return True
        if code in self.fallback.ENABLED_ERRORS:
            return True
        if code in self.fallback.DISABLED_ERRORS:
            return False
        return option.default_value


def parse_config_file(path: Path) -> Iterable[ConfigOption]:
    with path.open("rb") as f:
        data = tomli.load(f)
    data = data.get("tool", {}).get("pyanalyze", {})
    yield from _parse_config_section(data, path=path)


def _parse_config_section(
    section: Mapping[str, Any], module_path: ModulePath = (), *, path: Path
) -> Iterable[ConfigOption]:
    if "module" in section:
        if module_path == ():
            raise InvalidConfigOption(
                "Top-level configuration should not set module option"
            )
    for key, value in section.items():
        if key == "module":
            if module_path == ():
                raise InvalidConfigOption(
                    "Top-level configuration should not set module option"
                )
        elif key == "overrides":
            if not isinstance(value, (list, tuple)):
                raise InvalidConfigOption("overrides section must be a list")
            for override in value:
                if not isinstance(override, dict):
                    raise InvalidConfigOption("override value must be a dict")
                if "module" not in override or not isinstance(override["module"], str):
                    raise InvalidConfigOption(
                        "override section must set 'module' to a string"
                    )
                override_path = tuple(override["module"].split("."))
                yield from _parse_config_section(override, override_path, path=path)
        else:
            try:
                option_cls = ConfigOption.registry[key]
            except KeyError:
                raise InvalidConfigOption(f"Invalid configuration option {key!r}")
            yield option_cls(option_cls.parse(value, path), module_path)
