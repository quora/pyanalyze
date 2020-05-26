"""

Runs pyanalyze on itself.

"""
import os.path
import pyanalyze
from pyanalyze.error_code import ErrorCode
from pyanalyze.test_node_visitor import skip_before


class PyanalyzeConfig(pyanalyze.config.Config):
    DEFAULT_DIRS = (str(os.path.dirname(__file__)),)
    DEFAULT_BASE_MODULE = pyanalyze
    ENABLED_ERRORS = {
        ErrorCode.condition_always_true,
        ErrorCode.possibly_undefined_name,
    }


class PyanalyzeVisitor(pyanalyze.name_check_visitor.NameCheckVisitor):
    config = PyanalyzeConfig()
    should_check_environ_for_files = False


# on 2.7 it complains about too many things that exist only in Python 3
@skip_before((3, 6))
def test_all():
    PyanalyzeVisitor.check_all_files()


if __name__ == "__main__":
    PyanalyzeVisitor.main()
