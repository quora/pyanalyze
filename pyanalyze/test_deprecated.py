# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import AnySource, AnyValue, KnownValue, SubclassValue, TypedValue


class TestOverload(TestNameCheckVisitorBase):
    @assert_passes()
    def test_stub(self):
        def capybara():
            from _pyanalyze_tests.deprecated import deprecated_overload, y

            deprecated_overload(1)  # E: deprecated
            deprecated_overload("x")

    @assert_passes()
    def test_runtime(self):
        from pyanalyze.extensions import deprecated, overload

        @overload
        @deprecated("int support is deprecated")
        def deprecated_overload(x: int) -> int:
            ...

        @overload
        def deprecated_overload(x: str) -> str:
            ...

        def deprecated_overload(x):
            return x

        def capybara():
            deprecated_overload(1)  # E: deprecated
            deprecated_overload("x")
