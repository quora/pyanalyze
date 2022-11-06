# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import TypedValue


class TestLiteralString(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing_extensions import LiteralString

        def f(x: LiteralString) -> LiteralString:
            return "x"

        def capybara(x: str, y: LiteralString):
            f(x)  # E: incompatible_argument
            f(y)
            f("x")
            assert_is_value(f("x"), TypedValue(str, literal_only=True))
