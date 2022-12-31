# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestRecursion(TestNameCheckVisitorBase):
    @assert_passes()
    def test_runtime(self):
        from typing import Dict, List, Union

        JSON = Union[Dict[str, "JSON"], List["JSON"], int, str, float, bool, None]

        def f(x: JSON):
            pass

        def capybara() -> None:
            f([])
            f(b"x")  # E: incompatible_argument

    @assert_passes()
    def test_stub(self):
        def capybara():
            from _pyanalyze_tests.recursion import StrJson

            def want_str(cm: StrJson) -> None:
                pass

            def f(x: str):
                want_str(x)
                want_str(3)  # E: incompatible_argument
