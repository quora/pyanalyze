"""

Defines some concrete options that cannot easily be placed elsewhere.

"""
from pathlib import Path
from typing import Sequence, Callable
from types import ModuleType

from .config import Config
from .error_code import ErrorCode, DISABLED_BY_DEFAULT, ERROR_DESCRIPTION
from .options import PathSequenceOption, BooleanOption, PyObjectSequenceOption
from .value import VariableNameValue


class Paths(PathSequenceOption):
    """Paths that pyanalyze should type check."""

    name = "paths"
    is_global = True
    should_create_command_line_option = False

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[Path]:
        return [Path(s) for s in fallback.DEFAULT_DIRS]


class ImportPaths(PathSequenceOption):
    """Directories that pyanalyze may import from."""

    name = "import_paths"
    is_global = True


class EnforceNoUnused(BooleanOption):
    """If True, an error is raised when pyanalyze finds any unused objects."""

    name = "enforce_no_unused"
    is_global = True

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> bool:
        return fallback.ENFORCE_NO_UNUSED_OBJECTS


class VariableNameValues(PyObjectSequenceOption[VariableNameValue]):
    """List of :class:`pyanalyze.value.VariableNameValue` instances that create pseudo-types
    associated with certain variable names."""

    name = "variable_name_values"
    is_global = True

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[VariableNameValue]:
        return fallback.VARIABLE_NAME_VALUES


for _code in ErrorCode:
    type(
        _code.name,
        (BooleanOption,),
        {
            "__doc__": ERROR_DESCRIPTION[_code],
            "name": _code.name,
            "default_value": _code not in DISABLED_BY_DEFAULT,
            "should_create_command_line_option": False,
        },
    )

_IgnoreUnusedFunc = Callable[[ModuleType, str, object], bool]


class IgnoreUnused(PyObjectSequenceOption[_IgnoreUnusedFunc]):
    """If any of these functions returns True, we will exclude this
    object from the unused object check.

    The arguments are the module the object was found in, the attribute used to
    access it, and the object itself.

    """

    name = "ignore_unused"

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[_IgnoreUnusedFunc]:
        return [fallback.should_ignore_unused]
