# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, only_before
from .tests import make_simple_sequence
from .value import (
    AnySource,
    AnyValue,
    GenericValue,
    KnownValue,
    SequenceValue,
    TypedValue,
)


class TestGreaterLesser(TestNameCheckVisitorBase):
    @assert_passes()
    def test_gt(self):
        from typing_extensions import Annotated
        from typing import Any
        from annotated_types import Gt, Ge

        def takes_gt_5(x: Annotated[Any, Gt(5)]) -> None:
            pass

        def capybara(
            i: int,
            unnannotated,
            gt_4: Annotated[int, Gt(4)],
            gt_5: Annotated[int, Gt(5)],
            gt_6: Annotated[int, Gt(6)],
            ge_5: Annotated[int, Ge(5)],
            ge_6: Annotated[int, Ge(6)],
        ) -> None:
            takes_gt_5(i)  # E: incompatible_argument
            takes_gt_5(unnannotated)
            takes_gt_5(gt_4)  # E: incompatible_argument
            takes_gt_5(gt_5)
            takes_gt_5(gt_6)
            takes_gt_5(ge_5)  # E: incompatible_argument
            takes_gt_5(ge_6)
            takes_gt_5(5)  # E: incompatible_argument
            takes_gt_5(6)
            takes_gt_5("not an int")  # E: incompatible_argument
