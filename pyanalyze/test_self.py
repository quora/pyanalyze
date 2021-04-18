"""

Runs pyanalyze on itself.

"""
import os.path
import pyanalyze
from pyanalyze.error_code import ErrorCode


class PyanalyzeConfig(pyanalyze.config.Config):
    DEFAULT_DIRS = (str(os.path.dirname(__file__)),)
    DEFAULT_BASE_MODULE = pyanalyze
    ENFORCE_NO_UNUSED_OBJECTS = True
    ENABLED_ERRORS = {
        ErrorCode.condition_always_true,
        ErrorCode.possibly_undefined_name,
        ErrorCode.use_fstrings,
        ErrorCode.missing_return_annotation,
        ErrorCode.missing_parameter_annotation,
    }


class PyanalyzeVisitor(pyanalyze.name_check_visitor.NameCheckVisitor):
    config = PyanalyzeConfig()
    should_check_environ_for_files = False


def test_all() -> None:
    PyanalyzeVisitor.check_all_files()


if __name__ == "__main__":
    PyanalyzeVisitor.main()
