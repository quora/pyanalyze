"""

Runs pyanalyze on itself.

"""
import pyanalyze


class PyanalyzeVisitor(pyanalyze.name_check_visitor.NameCheckVisitor):
    should_check_environ_for_files = False
    config_filename = "../pyproject.toml"


def test_all() -> None:
    PyanalyzeVisitor.check_all_files()


if __name__ == "__main__":
    PyanalyzeVisitor.main()
