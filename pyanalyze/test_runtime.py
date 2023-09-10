from typing import List

from .runtime import is_compatible, get_compatibility_error


def test_is_compatible() -> None:
    assert not is_compatible(List[int], 42)
    assert is_compatible(List[int], [])
    assert not is_compatible(List[int], ["x"])
    assert is_compatible(List[int], [1, 2, 3])


def test_get_compatibility_error() -> None:
    assert (
        get_compatibility_error(List[int], 42) == "Cannot assign Literal[42] to list\n"
    )
    assert get_compatibility_error(List[int], []) is None
    assert (
        get_compatibility_error(List[int], ["x"])
        == "In element 0\n  Cannot assign Literal['x'] to int\n"
    )
    assert get_compatibility_error(List[int], [1, 2, 3]) is None
