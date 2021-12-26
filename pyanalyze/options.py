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
import qcore
import tomli
from unittest import mock

from .find_unused import used
from .config import Config
from .error_code import ErrorCode, DISABLED_BY_DEFAULT, ERROR_DESCRIPTION
from .safe import safe_in

T = TypeVar("T")
OptionT = TypeVar("OptionT", bound="ConfigOption")
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

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> T:
        raise NotFound

    @classmethod
    def get_fallback_option(cls: Type[OptionT], fallback: Config) -> Optional[OptionT]:
        try:
            val = cls.get_value_from_fallback(fallback)
        except NotFound:
            return None
        else:
            return cls(val)


class BooleanOption(ConfigOption[bool]):
    default_value = False

    @classmethod
    def parse(cls: "Type[BooleanOption]", data: object, source_path: Path) -> bool:
        if isinstance(data, bool):
            return data
        raise InvalidConfigOption.from_parser(cls, "bool", data)


class ConcatenatedOption(ConfigOption[Sequence[T]]):
    """Option for which the value is the concatenation of all the overrides."""

    @classmethod
    def get_value_from_instances(
        cls: "Type[ConcatenatedOption[T]]",
        instances: Sequence["ConcatenatedOption[T]"],
        module_path: ModulePath,
    ) -> Sequence[T]:
        # TODO after we clean up the fallback logic, this should
        # automatically incorporate the default value too.
        values = []
        for instance in instances:
            if instance.is_applicable_to(module_path):
                values += instance.value
        return values


class StringSequenceOption(ConcatenatedOption[str]):
    @classmethod
    def parse(
        cls: "Type[StringSequenceOption]", data: object, source_path: Path
    ) -> Sequence[str]:
        if isinstance(data, (list, tuple)) and all(
            isinstance(elt, str) for elt in data
        ):
            return data
        raise InvalidConfigOption.from_parser(cls, "sequence of strings", data)


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


class PyObjectSequenceOption(ConfigOption[Sequence[T]]):
    """Represents a sequence of objects parsed as Python objects."""

    default_value = ()

    @classmethod
    def parse(
        cls: "Type[PyObjectSequenceOption[T]]", data: object, source_path: Path
    ) -> Sequence[T]:
        if not isinstance(data, (list, tuple)):
            raise InvalidConfigOption.from_parser(
                cls, "sequence of Python objects", data
            )
        final = []
        for elt in data:
            try:
                obj = qcore.object_from_string(elt)
            except Exception:
                raise InvalidConfigOption.from_parser(
                    cls, "path to Python object", elt
                ) from None
            used(obj)
            final.append(elt)
        return final

    @classmethod
    def contains(cls, obj: object, options: "Options") -> bool:
        val = options.get_value_for(cls)
        return safe_in(obj, val)


class Paths(PathSequenceOption):
    """Paths that pyanalyze should type check."""

    name = "paths"
    is_global = True

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[Path]:
        return [Path(s) for s in fallback.DEFAULT_DIRS]


class ImportPaths(PathSequenceOption):
    """Directories that pyanalyze may import from."""

    name = "import_paths"
    is_global = True


class IgnoredPaths(ConcatenatedOption[Sequence[str]]):
    """Attribute accesses on these do not result in errors."""

    name = "ignored_paths"
    default_value = ()

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[Sequence[str]]:
        return fallback.IGNORED_PATHS

    @classmethod
    def parse(cls, data: object, source_path: Path) -> Sequence[Sequence[str]]:
        if not isinstance(data, (list, tuple)):
            raise InvalidConfigOption.from_parser(cls, "sequence", data)
        for sublist in data:
            if not isinstance(sublist, (list, tuple)):
                raise InvalidConfigOption.from_parser(cls, "sequence", sublist)
            for elt in sublist:
                if not isinstance(elt, str):
                    raise InvalidConfigOption.from_parser(cls, "string", elt)
        return data


class IgnoredEndOfReference(StringSequenceOption):
    """When these attributes are accessed but they don't exist, the error is ignored."""

    name = "ignored_end_of_reference"
    default_value = [
        # these are created by the mock module
        "call_count",
        "assert_has_calls",
        "reset_mock",
        "called",
        "assert_called_once",
        "assert_called_once_with",
        "assert_called_with",
        "count",
        "assert_any_call",
        "assert_not_called",
    ]

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[str]:
        return list(fallback.IGNORED_END_OF_REFERENCE)


class ExtraBuiltins(StringSequenceOption):
    """Even if these variables are undefined, no errors are shown."""

    name = "extra_builtins"
    default_value = ["__IPYTHON__"]  # special global defined in IPython

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[str]:
        return list(fallback.IGNORED_VARIABLES)


class UnimportableModules(StringSequenceOption):
    """Do not attempt to import these modules if they are imported within a function."""

    default_value = []
    name = "unimportable_modules"

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[str]:
        return list(fallback.UNIMPORTABLE_MODULES)


class EnforceNoUnused(BooleanOption):
    """If True, an error is raised when pyanalyze finds any unused objects."""

    name = "enforce_no_unused"
    is_global = True

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> bool:
        return fallback.ENFORCE_NO_UNUSED_OBJECTS


class IgnoreNoneAttributes(BooleanOption):
    """If True, we ignore None when type checking attribute access on a Union
    type."""

    name = "ignore_none_attributes"

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> bool:
        return fallback.IGNORE_NONE_ATTRIBUTES


class ForLoopAlwaysEntered(BooleanOption):
    """If True, we assume that for loops are always entered at least once,
    which affects the potentially_undefined_name check. This will miss
    some bugs but also remove some annoying false positives."""

    name = "for_loop_always_entered"

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> bool:
        return fallback.FOR_LOOP_ALWAYS_ENTERED


class IgnoredCallees(PyObjectSequenceOption[object]):
    """Calls to these aren't checked for argument validity."""

    default_value = [
        # getargspec gets confused about this subclass of tuple that overrides __new__ and __call__
        mock.call,
        mock.MagicMock,
        mock.Mock,
    ]
    name = "ignored_callees"

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[object]:
        return fallback.IGNORED_CALLEES


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
        try:
            return self._get_value_for_no_default(option)
        except NotFound:
            return option.default_value

    def _get_value_for_no_default(self, option: Type[ConfigOption[T]]) -> T:
        instances = self.options.get(option.name, ())
        fallback = option.get_fallback_option(self.fallback)
        if fallback is not None:
            instances = [*instances, fallback]
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
