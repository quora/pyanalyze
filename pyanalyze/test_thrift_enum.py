# static analysis: ignore

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import KnownValue, TypedValue, assert_is_value


class TestThriftEnum(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        class ThriftEnum(object):
            X = 0
            Y = 1

            _VALUES_TO_NAMES = {0: "X", 1: "Y"}
            _NAMES_TO_VALUES = {"X": 0, "Y": 1}

        def want_enum(e: ThriftEnum):
            pass

        def capybara(e: ThriftEnum):
            want_enum(e)
            want_enum(ThriftEnum.X)
            want_enum(ThriftEnum.Y)
            want_enum(0)
            want_enum(1)
            want_enum(42)  # E: incompatible_argument
            want_enum(str(e))  # E: incompatible_argument

    @assert_passes()
    def test_typevar(self):
        from typing import TypeVar
        from typing_extensions import Annotated

        class ThriftEnum(object):
            X = 0
            Y = 1

            _VALUES_TO_NAMES = {0: "X", 1: "Y"}
            _NAMES_TO_VALUES = {"X": 0, "Y": 1}

        TET = TypeVar("TET", bound=ThriftEnum)

        def want_enum(te: ThriftEnum) -> None:
            pass

        def get_it(te: TET) -> TET:
            want_enum(te)
            return te

        def get_it_annotated(te: Annotated[TET, 3]) -> TET:
            want_enum(te)
            return te

        def capybara(e: ThriftEnum):
            assert_is_value(get_it(e), TypedValue(ThriftEnum))
            assert_is_value(get_it(ThriftEnum.X), KnownValue(ThriftEnum.X))

            assert_is_value(get_it_annotated(e), TypedValue(ThriftEnum))
            assert_is_value(get_it_annotated(ThriftEnum.X), KnownValue(ThriftEnum.X))
