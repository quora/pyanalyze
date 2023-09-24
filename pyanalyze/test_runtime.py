# static analysis: ignore
from typing import List

from .runtime import is_compatible, get_compatibility_error
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


def test_is_compatible() -> None:
    assert not is_compatible(42, List[int])
    assert is_compatible([], List[int])
    assert not is_compatible(["x"], List[int])
    assert is_compatible([1, 2, 3], List[int])


def test_get_compatibility_error() -> None:
    assert (
        get_compatibility_error(42, List[int]) == "Cannot assign Literal[42] to list\n"
    )
    assert get_compatibility_error([], List[int]) is None
    assert (
        get_compatibility_error(["x"], List[int])
        == "In element 0\n  Cannot assign Literal['x'] to int\n"
    )
    assert get_compatibility_error([1, 2, 3], List[int]) is None


class TestRuntimeTypeGuard(TestNameCheckVisitorBase):
    @assert_passes()
    def test_runtime(self):
        from typing_extensions import Annotated
        from annotated_types import Predicate
        from pyanalyze.runtime import is_compatible

        IsLower = Annotated[str, Predicate(str.islower)]

        def want_lowercase(s: IsLower) -> None:
            assert s.islower()

        def capybara(s: str) -> None:
            want_lowercase(s)  # E: incompatible_argument
            if is_compatible(s, IsLower):
                want_lowercase(s)

        def asserting_capybara(s: str) -> None:
            assert is_compatible(s, IsLower)
            want_lowercase(s)
