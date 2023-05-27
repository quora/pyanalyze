# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import KnownValue


class TestInferenceHelpers(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        from pyanalyze import assert_is_value, dump_value
        from pyanalyze.value import Value

        def capybara(val: Value) -> None:
            reveal_type(dump_value)  # E: reveal_type
            dump_value(reveal_type)  # E: reveal_type
            assert_is_value(1, KnownValue(1))
            assert_is_value(1, KnownValue(2))  # E: inference_failure
            assert_is_value(1, val)  # E: inference_failure

    @assert_passes()
    def test_return_value(self) -> None:
        from pyanalyze import assert_is_value, dump_value

        def capybara():
            x = dump_value(1)  # E: reveal_type
            y = reveal_type(1)  # E: reveal_type
            assert_is_value(x, KnownValue(1))
            assert_is_value(y, KnownValue(1))

    @assert_passes()
    def test_assert_type(self) -> None:
        from typing import Any

        from pyanalyze.extensions import assert_type

        def capybara(x: int) -> None:
            assert_type(x, int)
            assert_type(x, "int")
            assert_type(x, Any)  # E: inference_failure
            assert_type(x, str)  # E: inference_failure


class TestAssertError(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        from pyanalyze.extensions import assert_error

        def f(x: int) -> None:
            pass

        def capybara() -> None:
            with assert_error():
                f("x")

            with assert_error():  # E: inference_failure
                f(1)


class TestRevealLocals(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        from pyanalyze.extensions import reveal_locals

        def capybara(a: object, b: str) -> None:
            c = 3
            if b == "x":
                reveal_locals()  # E: reveal_type
            print(a, b, c)
