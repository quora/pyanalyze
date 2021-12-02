# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import GenericValue, TypedValue, assert_is_value


class TestSyntheticType(TestNameCheckVisitorBase):
    @assert_passes()
    def test_functools(self):
        import functools

        def f() -> int:
            return 0

        def capybara():
            c = functools.singledispatch(f)
            assert_is_value(
                c, GenericValue("functools._SingleDispatchCallable", [TypedValue(int)])
            )
