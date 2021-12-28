# static analysis: ignore
from .value import TypedValue, assert_is_value
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


class TestTypeEvaluation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_is_set(self):
        from pyanalyze.test_type_evaluation import simple_evaluated

        def capybara():
            assert_is_value(simple_evaluated(1), TypedValue(str))
            assert_is_value(simple_evaluated(1, "1"), TypedValue(int))
