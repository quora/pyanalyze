# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestSysPlatform(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        import os
        import sys

        from typing_extensions import assert_type

        def capybara() -> None:
            if sys.platform == "win32":
                assert_type(os.P_DETACH, int)
            else:
                os.P_DETACH  # E: undefined_attribute


class TestSysVersion(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        import ast
        import sys

        from typing_extensions import assert_type

        if sys.version_info >= (3, 10):

            def capybara(m: ast.Match) -> None:
                assert_type(m, ast.Match)

        if sys.version_info >= (3, 12):

            def pacarana(m: ast.TypeVar) -> None:
                assert_type(m, ast.TypeVar)
