# static analysis: ignore
from typing_extensions import Literal

from .value import AnySource, AnyValue, TypedValue, assert_is_value
from .test_node_visitor import assert_passes
from .test_name_check_visitor import TestNameCheckVisitorBase
from .extensions import evaluated, is_keyword, is_positional, is_provided, is_of_type

from typing import Union


# These functions must be defined globally because the
# mechanism for finding the source code doesn't work inside
# tests.
@evaluated
def simple_evaluated(x: int, y: str = ""):
    if is_provided(y):
        return int
    else:
        return str


def simple_evaluated(*args: object) -> Union[int, str]:
    if len(args) >= 2:
        return 1
    else:
        return "x"


@evaluated
def is_of_type_evaluated(x: int):
    if is_of_type(x, Literal[1]):
        return str
    else:
        return int


def is_of_type_evaluated(x: int) -> Union[int, str]:
    if x == 1:
        return ""
    else:
        return 0


@evaluated
def not_evaluated(x: int):
    if not is_of_type(x, Literal[1]):
        return str
    else:
        return int


def not_evaluated(x: int) -> Union[int, str]:
    if x != 1:
        return ""
    else:
        return 0


@evaluated
def compare_evaluated(x: object):
    if x is None:
        return str
    elif x == 1:
        return float
    else:
        return int


def compare_evaluated(x: object) -> Union[int, str, float]:
    raise NotImplementedError


@evaluated
def nonempty_please(x: str) -> int:
    if x == "":
        raise Exception("Non-empty string expected")
    else:
        return int


def nonempty_please(x: str) -> int:
    assert x
    return len(x)


@evaluated
def restrict_kind(x: str, y: int):
    if is_keyword(x):
        raise Exception("x must be positional")
    if is_positional(y):
        raise Exception("y must be keyword")
    return int


def restrict_kind(*args, **kwargs):
    return 0


class TestTypeEvaluation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_is_provided(self):
        from pyanalyze.test_type_evaluation import simple_evaluated

        def capybara():
            assert_is_value(simple_evaluated(1), TypedValue(str))
            assert_is_value(simple_evaluated(1, "1"), TypedValue(int))

    @assert_passes()
    def test_is_of_type(self):
        from pyanalyze.test_type_evaluation import is_of_type_evaluated

        def capybara(unannotated):
            assert_is_value(is_of_type_evaluated(1), TypedValue(str))
            assert_is_value(is_of_type_evaluated(2), TypedValue(int))
            assert_is_value(
                is_of_type_evaluated(unannotated),
                AnyValue(AnySource.multiple_overload_matches),
            )
            assert_is_value(
                is_of_type_evaluated(2 if unannotated else 1),
                TypedValue(int) | TypedValue(str),
            )

    @assert_passes()
    def test_not(self):
        from pyanalyze.test_type_evaluation import not_evaluated

        def capybara(unannotated):
            assert_is_value(not_evaluated(1), TypedValue(int))
            assert_is_value(not_evaluated(2), TypedValue(str))
            assert_is_value(
                not_evaluated(unannotated),
                AnyValue(AnySource.multiple_overload_matches),
            )
            assert_is_value(
                not_evaluated(2 if unannotated else 1),
                TypedValue(int) | TypedValue(str),
            )

    @assert_passes()
    def test_compare(self):
        from pyanalyze.test_type_evaluation import compare_evaluated

        def capybara(unannotated):
            assert_is_value(compare_evaluated(None), TypedValue(str))
            assert_is_value(compare_evaluated(1), TypedValue(float))
            assert_is_value(compare_evaluated("x"), TypedValue(int))
            assert_is_value(
                compare_evaluated(None if unannotated else 1),
                TypedValue(str) | TypedValue(float),
            )

    @assert_passes()
    def test_error(self):
        from pyanalyze.test_type_evaluation import nonempty_please

        def capybara():
            nonempty_please("")  # E: incompatible_call
            assert_is_value(nonempty_please("x"), TypedValue(int))

    @assert_passes()
    def test_restrict_kind(self):
        from pyanalyze.test_type_evaluation import restrict_kind

        def capybara(stuff):
            restrict_kind("x", y=1)
            restrict_kind(x="x", y=1)  # E: incompatible_call
            restrict_kind("x", 1)  # E: incompatible_call
            restrict_kind(*stuff, **stuff)
            restrict_kind(**stuff)  # E: incompatible_call
            restrict_kind(*stuff)  # E: incompatible_call
