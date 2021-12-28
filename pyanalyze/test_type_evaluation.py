# static analysis: ignore
from typing_extensions import Literal
from .value import AnySource, AnyValue, TypedValue, assert_is_value
from .test_node_visitor import assert_passes
from .test_name_check_visitor import TestNameCheckVisitorBase
from .extensions import evaluated, is_set

from typing import Union


# These functions must be defined globally because the
# mechanism for finding the source code doesn't work inside
# tests.
@evaluated
def simple_evaluated(x: int, y: str = ""):
    if is_set(y):
        return int
    else:
        return str


def simple_evaluated(*args: object) -> Union[int, str]:
    if len(args) >= 2:
        return 1
    else:
        return "x"


@evaluated
def isinstance_evaluated(x: int):
    if isinstance(x, Literal[1]):
        return str
    else:
        return int


def isinstance_evaluated(x: int) -> Union[int, str]:
    if x == 1:
        return ""
    else:
        return 0


class TestTypeEvaluation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_is_set(self):
        from pyanalyze.test_type_evaluation import simple_evaluated

        def capybara():
            assert_is_value(simple_evaluated(1), TypedValue(str))
            assert_is_value(simple_evaluated(1, "1"), TypedValue(int))

    @assert_passes()
    def test_isinstance(self):
        from pyanalyze.test_type_evaluation import isinstance_evaluated

        def capybara(unannotated):
            assert_is_value(isinstance_evaluated(1), TypedValue(str))
            assert_is_value(isinstance_evaluated(2), TypedValue(int))
            assert_is_value(
                isinstance_evaluated(unannotated),
                AnyValue(AnySource.multiple_overload_matches),
            )
            assert_is_value(
                isinstance_evaluated(2 if unannotated else 1),
                TypedValue(int) | TypedValue(str),
            )
