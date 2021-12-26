"""

Defines some concrete options that cannot easily be placed elsewhere.

"""
from pathlib import Path
from typing import Sequence

from .config import Config
from .error_code import ErrorCode, DISABLED_BY_DEFAULT, ERROR_DESCRIPTION
from .options import PathSequenceOption, BooleanOption


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


class EnforceNoUnused(BooleanOption):
    """If True, an error is raised when pyanalyze finds any unused objects."""

    name = "enforce_no_unused"
    is_global = True

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> bool:
        return fallback.ENFORCE_NO_UNUSED_OBJECTS


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
