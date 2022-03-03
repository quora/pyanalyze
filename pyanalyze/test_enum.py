# static analysis: ignore
from .implementation import assert_is_value
from .value import SubclassValue, TypedValue
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestEnum(TestNameCheckVisitorBase):
    @assert_passes()
    def test_functional(self):
        from enum import Enum, IntEnum

        def capybara():
            X = Enum("X", ["a", "b", "c"])
            assert_is_value(X, SubclassValue(TypedValue(Enum)))

            IE = IntEnum("X", ["a", "b", "c"])
            assert_is_value(IE, SubclassValue(TypedValue(Enum)))

    @assert_passes()
    def test_call(self):
        from enum import Enum

        class X(Enum):
            a = 1
            b = 2

        def capybara():
            assert_is_value(X(1), TypedValue(X))
            # This should be an error, but the typeshed
            # stubs are too lenient.
            assert_is_value(X(None), TypedValue(X))
