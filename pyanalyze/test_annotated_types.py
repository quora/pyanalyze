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

    @assert_passes()
    def test_ge(self):
        from typing_extensions import Annotated
        from typing import Any
        from annotated_types import Ge, Gt

        def takes_ge_5(x: Annotated[Any, Ge(5)]) -> None:
            pass

        def capybara(
            i: int,
            unnannotated,
            ge_4: Annotated[int, Ge(4)],
            ge_5: Annotated[int, Ge(5)],
            ge_6: Annotated[int, Ge(6)],
            gt_4: Annotated[int, Gt(4)],
            gt_5: Annotated[int, Gt(5)],
        ) -> None:
            takes_ge_5(i)  # E: incompatible_argument
            takes_ge_5(unnannotated)
            takes_ge_5(ge_4)  # E: incompatible_argument
            takes_ge_5(ge_5)
            takes_ge_5(ge_6)
            # possibly should be allowed, but what if it's a float?
            takes_ge_5(gt_4)  # E: incompatible_argument
            takes_ge_5(gt_5)
            takes_ge_5(4)  # E: incompatible_argument
            takes_ge_5(5)
            takes_ge_5(6)

    @assert_passes()
    def test_lt(self):
        from typing_extensions import Annotated
        from typing import Any
        from annotated_types import Lt, Le

        def takes_lt_5(x: Annotated[Any, Lt(5)]) -> None:
            pass

        def capybara(
            i: int,
            unnannotated,
            lt_4: Annotated[int, Lt(4)],
            lt_5: Annotated[int, Lt(5)],
            lt_6: Annotated[int, Lt(6)],
            le_4: Annotated[int, Le(4)],
            le_5: Annotated[int, Le(5)],
        ) -> None:
            takes_lt_5(i)  # E: incompatible_argument
            takes_lt_5(unnannotated)
            takes_lt_5(lt_4)
            takes_lt_5(lt_5)
            takes_lt_5(lt_6)  # E: incompatible_argument
            takes_lt_5(le_4)
            takes_lt_5(le_5)  # E: incompatible_argument
            takes_lt_5(4)
            takes_lt_5(5)  # E: incompatible_argument

    @assert_passes()
    def test_le(self):
        from typing_extensions import Annotated
        from typing import Any
        from annotated_types import Le, Lt

        def takes_le_5(x: Annotated[Any, Le(5)]) -> None:
            pass

        def capybara(
            i: int,
            unnannotated,
            le_4: Annotated[int, Le(4)],
            le_5: Annotated[int, Le(5)],
            le_6: Annotated[int, Le(6)],
            lt_5: Annotated[int, Lt(5)],
            lt_6: Annotated[int, Lt(6)],
        ) -> None:
            takes_le_5(i)  # E: incompatible_argument
            takes_le_5(unnannotated)
            takes_le_5(le_4)
            takes_le_5(le_5)
            takes_le_5(le_6)  # E: incompatible_argument
            takes_le_5(lt_5)
            # possibly should be allowed, but what if it's a float?
            takes_le_5(lt_6)  # E: incompatible_argument
            takes_le_5(4)
            takes_le_5(5)
            takes_le_5(6)  # E: incompatible_argument
