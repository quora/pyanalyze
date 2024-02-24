# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestAnnotatedTypesAnnotations(TestNameCheckVisitorBase):
    @assert_passes()
    def test_gt(self):
        from typing import Any

        from annotated_types import Ge, Gt
        from typing_extensions import Annotated

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
        from typing import Any

        from annotated_types import Ge, Gt
        from typing_extensions import Annotated

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
        from typing import Any

        from annotated_types import Le, Lt
        from typing_extensions import Annotated

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
        from typing import Any

        from annotated_types import Le, Lt
        from typing_extensions import Annotated

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

    @assert_passes()
    def test_multiple_of(self):
        from annotated_types import MultipleOf
        from typing_extensions import Annotated

        def takes_multiple_of_10(x: Annotated[int, MultipleOf(10)]) -> None:
            pass

        def capybara(
            i: int,
            unannotated,
            multiple_of_2: Annotated[int, MultipleOf(2)],
            multiple_of_10: Annotated[int, MultipleOf(10)],
            multiple_of_20: Annotated[int, MultipleOf(20)],
        ) -> None:
            takes_multiple_of_10(i)  # E: incompatible_argument
            takes_multiple_of_10(unannotated)
            takes_multiple_of_10(multiple_of_2)  # E: incompatible_argument
            takes_multiple_of_10(multiple_of_10)
            takes_multiple_of_10(multiple_of_20)
            takes_multiple_of_10(2)  # E: incompatible_argument
            takes_multiple_of_10(10)
            takes_multiple_of_10(50)

    @assert_passes()
    def test_min_max_len(self):
        from annotated_types import Len, MaxLen, MinLen
        from typing_extensions import Annotated, NotRequired, TypedDict

        def takes_min_len_5(x: Annotated[object, MinLen(5)]) -> None:
            pass

        def takes_max_len_5(x: Annotated[object, MaxLen(5)]) -> None:
            pass

        def takes_3_to_5(x: Annotated[object, Len(3, 5)]) -> None:
            pass

        class TD3to5(TypedDict):
            a: int
            b: int
            c: int
            d: NotRequired[int]
            e: NotRequired[int]

        def capybara(
            i: int,
            min_4: Annotated[str, MinLen(4)],
            min_6: Annotated[str, MinLen(6)],
            max_4: Annotated[str, MaxLen(4)],
            max_6: Annotated[str, MaxLen(6)],
            exactly_5: Annotated[str, Len(5, 5)],
            td: TD3to5,
            unannotated,
        ) -> None:
            takes_min_len_5((1, 2, 3, 4))  # E: incompatible_argument
            takes_min_len_5((1, 2, 3, 4, 5))
            takes_min_len_5((1, 2, 3, 4, 5, 6))
            takes_min_len_5((i, i, i, i))  # E: incompatible_argument
            takes_min_len_5((i, i, i, i, i))
            takes_min_len_5(min_4)  # E: incompatible_argument
            takes_min_len_5(min_6)
            takes_min_len_5(exactly_5)
            takes_min_len_5(td)  # E: incompatible_argument
            takes_min_len_5((i, i, i, i, i, *unannotated))

            takes_max_len_5((1, 2, 3, 4, 5))
            takes_max_len_5((1, 2, 3, 4, 5, 6))  # E: incompatible_argument
            takes_max_len_5((i, i, i, i, i))
            takes_max_len_5((i, i, i, i, i, i))  # E: incompatible_argument
            takes_max_len_5(max_4)
            takes_max_len_5(max_6)  # E: incompatible_argument
            takes_max_len_5(exactly_5)
            # TypedDict don't satisfy MaxLen as they're allowed to have
            # arbitrary additional keys at runtime
            takes_max_len_5(td)  # E: incompatible_argument
            takes_max_len_5((i, *unannotated))  # E: incompatible_argument

            takes_3_to_5((1, 2))  # E: incompatible_argument
            takes_3_to_5((1, 2, 3))
            takes_3_to_5(exactly_5)
            takes_3_to_5(td)  # E: incompatible_argument

    @assert_passes()
    def test_timezone(self):
        from datetime import datetime, timedelta, timezone

        from annotated_types import Timezone
        from typing_extensions import Annotated

        def takes_naive(x: Annotated[datetime, Timezone(None)]) -> None:
            pass

        def takes_aware(x: Annotated[datetime, Timezone(...)]) -> None:
            pass

        def takes_utc(x: Annotated[datetime, Timezone(timezone.utc)]) -> None:
            pass

        naive_dt = datetime.now()
        utc_dt = datetime.now(timezone.utc)
        non_utc_aware_dt = datetime.now(timezone(timedelta(hours=1)))

        def capybara(dt: datetime) -> None:
            takes_naive(dt)  # E: incompatible_argument
            takes_naive(naive_dt)
            takes_naive(utc_dt)  # E: incompatible_argument
            takes_naive(non_utc_aware_dt)  # E: incompatible_argument

            takes_aware(dt)  # E: incompatible_argument
            takes_aware(naive_dt)  # E: incompatible_argument
            takes_aware(utc_dt)
            takes_aware(non_utc_aware_dt)

            takes_utc(dt)  # E: incompatible_argument
            takes_utc(naive_dt)  # E: incompatible_argument
            takes_utc(utc_dt)
            takes_utc(non_utc_aware_dt)  # E: incompatible_argument

    @assert_passes()
    def test_predicate(self):
        from annotated_types import Predicate
        from typing_extensions import Annotated

        def takes_upper(x: Annotated[str, Predicate(str.isupper)]) -> None:
            pass

        def capybara(
            s: str, unannotated, scream: Annotated[str, Predicate(str.isupper)]
        ) -> None:
            takes_upper(s)  # E: incompatible_argument
            takes_upper(unannotated)
            takes_upper(scream)
            takes_upper("WHY DO YOU ONLY WANT UPPERCASE")
            takes_upper("lowercase")  # E: incompatible_argument


class TestInferAnnotations(TestNameCheckVisitorBase):
    @assert_passes()
    def test_infer_gt(self):
        from annotated_types import Gt
        from typing_extensions import Annotated

        def takes_gt_5(x: Annotated[int, Gt(5)]) -> None:
            pass

        def capybara(i: int) -> None:
            takes_gt_5(i)  # E: incompatible_argument

            if i > 4:
                takes_gt_5(i)  # E: incompatible_argument

            if i > 5:
                takes_gt_5(i)

            takes_gt_5(i)  # E: incompatible_argument

            if i > 6:
                takes_gt_5(i)

    @assert_passes()
    def test_len(self):
        from annotated_types import Len, MaxLen, MinLen
        from typing_extensions import Annotated

        def takes_len_5(x: Annotated[str, Len(5)]) -> None:
            pass

        def takes_min_len_5(x: Annotated[str, MinLen(5)]) -> None:
            pass

        def takes_max_len_5(x: Annotated[str, MaxLen(5)]) -> None:
            pass

        def capybara(s: str) -> None:
            if len(s) == 5:
                takes_len_5(s)
                takes_min_len_5(s)
                takes_max_len_5(s)

            if len(s) >= 6:
                takes_min_len_5(s)
                takes_max_len_5(s)  # E: incompatible_argument

            if len(s) <= 4:
                takes_max_len_5(s)
                takes_min_len_5(s)  # E: incompatible_argument

            if len(s) > 5:
                takes_min_len_5(s)
                takes_max_len_5(s)  # E: incompatible_argument

            if len(s) < 5:
                takes_max_len_5(s)
                takes_min_len_5(s)  # E: incompatible_argument
                takes_len_5(s)  # E: incompatible_argument
