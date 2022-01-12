"""

Structured configuration options.

"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import pathlib
from typing import (
    Any,
    ClassVar,
    Collection,
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

from .find_unused import used
from .config import Config
from .error_code import ErrorCode
from .safe import safe_in

try:
    from argparse import BooleanOptionalAction
except ImportError:
    # 3.8 and lower (modified from CPython)
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings: Sequence[str], **kwargs: Any) -> None:

            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)

                if option_string.startswith("--"):
                    option_string = "--no-" + option_string[2:]
                    _option_strings.append(option_string)

            super().__init__(option_strings=_option_strings, nargs=0, **kwargs)

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: object,
            option_string: Optional[str] = None,
        ) -> None:
            if option_string is not None and option_string in self.option_strings:
                setattr(namespace, self.dest, not option_string.startswith("--no-"))

        def format_usage(self) -> str:
            return " | ".join(self.option_strings)


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
    should_create_command_line_option: ClassVar[bool] = True
    value: T
    applicable_to: ModulePath = ()
    from_command_line: bool = False
    priority: int = 0  # higher number = lower priority

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
            self.priority,  # lower priority number first
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

    @classmethod
    def create_command_line_option(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError(cls)


class BooleanOption(ConfigOption[bool]):
    default_value = False

    @classmethod
    def parse(cls: "Type[BooleanOption]", data: object, source_path: Path) -> bool:
        if isinstance(data, bool):
            return data
        raise InvalidConfigOption.from_parser(cls, "bool", data)

    @classmethod
    def create_command_line_option(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            f"--{cls.name.replace('_', '-')}",
            action=BooleanOptionalAction,
            help=cls.__doc__,
            default=argparse.SUPPRESS,
        )


class IntegerOption(ConfigOption[int]):
    @classmethod
    def parse(cls: "Type[IntegerOption]", data: object, source_path: Path) -> int:
        if isinstance(data, int):
            return data
        raise InvalidConfigOption.from_parser(cls, "int", data)

    @classmethod
    def create_command_line_option(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            f"--{cls.name.replace('_', '-')}",
            type=int,
            help=cls.__doc__,
            default=argparse.SUPPRESS,
        )


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

    @classmethod
    def create_command_line_option(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            f"--{cls.name.replace('_', '-')}",
            action="append",
            help=cls.__doc__,
            default=argparse.SUPPRESS,
        )


class PathSequenceOption(ConfigOption[Sequence[Path]]):
    default_value: ClassVar[Sequence[Path]] = ()

    @classmethod
    def parse(
        cls: "Type[PathSequenceOption]", data: object, source_path: Path
    ) -> Sequence[str]:
        if isinstance(data, (list, tuple)) and all(
            isinstance(elt, str) for elt in data
        ):
            return [(source_path.parent / elt).resolve() for elt in data]
        raise InvalidConfigOption.from_parser(cls, "sequence of strings", data)

    @classmethod
    def create_command_line_option(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            f"--{cls.name.replace('_', '-')}",
            action="append",
            type=pathlib.Path,
            help=cls.__doc__,
            default=argparse.SUPPRESS,
        )


class PyObjectSequenceOption(ConcatenatedOption[T]):
    """Represents a sequence of objects parsed as Python objects."""

    default_value: ClassVar[Sequence[T]] = ()

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
            final.append(obj)
        return final

    @classmethod
    def contains(cls, obj: object, options: "Options") -> bool:
        val = options.get_value_for(cls)
        return safe_in(obj, val)

    @classmethod
    def create_command_line_option(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            f"--{cls.name.replace('_', '-')}",
            action="append",
            type=qcore.object_from_string,
            help=cls.__doc__,
            default=argparse.SUPPRESS,
        )


@dataclass
class Options:
    options: Mapping[str, Sequence[ConfigOption]]
    fallback: Config
    module_path: ModulePath = ()

    @classmethod
    def from_option_list(
        cls,
        instances: Sequence[ConfigOption] = (),
        fallback: Config = Config(),
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
        else:
            instances = [*instances, option(option.default_value)]
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

    def display(self) -> None:
        print("Options:")
        prefix = " " * 8
        for name, option_cls in sorted(ConfigOption.registry.items()):
            current_value = self.get_value_for(option_cls)
            print(f"    {name} (value: {current_value})")
            instances = self.options.get(name, [])
            for instance in instances:
                pieces = []
                if instance.applicable_to:
                    pieces.append(f"module: {'.'.join(instance.applicable_to)}")
                if instance.from_command_line:
                    pieces.append("from command line")
                else:
                    pieces.append("from config file")
                suffix = f" ({', '.join(pieces)})"
                print(f"{prefix}{instance.value}{suffix}")
        print(f"Fallback: {self.fallback}")
        if self.module_path:
            print(f"For module: {'.'.join(self.module_path)}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for cls in ConfigOption.registry.values():
        if not cls.should_create_command_line_option:
            continue
        cls.create_command_line_option(parser)


def parse_config_file(
    path: Path, *, priority: int = 0, seen_paths: Collection[Path] = frozenset()
) -> Iterable[ConfigOption]:
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError:
        raise InvalidConfigOption(f"Cannot open config file at {path}")
    if path in seen_paths:
        raise InvalidConfigOption("Recursive config inclusion detected")
    with path.open("rb") as f:
        data = tomli.load(f)
    data = data.get("tool", {}).get("pyanalyze", {})
    yield from _parse_config_section(
        data, path=path, priority=priority, seen_paths={path, *seen_paths}
    )


def _parse_config_section(
    section: Mapping[str, Any],
    module_path: ModulePath = (),
    *,
    path: Path,
    priority: int,
    seen_paths: Collection[Path],
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
        elif key == "extend_config":
            if not isinstance(value, str):
                raise InvalidConfigOption("extend_config must be a string")
            extended_path = path.parent / value
            yield from parse_config_file(
                extended_path, priority=priority + 1, seen_paths=seen_paths
            )
        elif key == "overrides":
            if module_path:
                raise InvalidConfigOption("Nested section cannot set overrides")
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
                yield from _parse_config_section(
                    override,
                    override_path,
                    path=path,
                    priority=priority,
                    seen_paths=seen_paths,
                )
        else:
            try:
                option_cls = ConfigOption.registry[key]
            except KeyError:
                raise InvalidConfigOption(f"Invalid configuration option {key!r}")
            yield option_cls(option_cls.parse(value, path), module_path)
