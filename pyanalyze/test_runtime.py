# static analysis: ignore
from typing import List

from .runtime import get_assignability_error, is_assignable
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


def test_is_assignable() -> None:
    assert not is_assignable(42, List[int])
    assert is_assignable([], List[int])
    assert not is_assignable(["x"], List[int])
    assert is_assignable([1, 2, 3], List[int])


def test_get_assignability_error() -> None:
    assert (
        get_assignability_error(42, List[int]) == "Cannot assign Literal[42] to list\n"
    )
    assert get_assignability_error([], List[int]) is None
    assert (
        get_assignability_error(["x"], List[int])
        == "In element 0\n  Cannot assign Literal['x'] to int\n"
    )
    assert get_assignability_error([1, 2, 3], List[int]) is None


class TestRuntimeTypeGuard(TestNameCheckVisitorBase):
    @assert_passes()
    def test_runtime(self):
        from annotated_types import Predicate
        from typing_extensions import Annotated

        from pyanalyze.runtime import is_assignable

        IsLower = Annotated[str, Predicate(str.islower)]

        def want_lowercase(s: IsLower) -> None:
            assert s.islower()

        def capybara(s: str) -> None:
            want_lowercase(s)  # E: incompatible_argument
            if is_assignable(s, IsLower):
                want_lowercase(s)

        def asserting_capybara(s: str) -> None:
            assert is_assignable(s, IsLower)
            want_lowercase(s)
