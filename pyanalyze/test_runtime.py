from .runtime import is_compatible, get_compatibility_error


def test_is_compatible() -> None:
    assert not is_compatible(list[int], 42)
    assert is_compatible(list[int], [])
    assert not is_compatible(list[int], ["x"])
    assert is_compatible(list[int], [1, 2, 3])


def test_get_compatibility_error() -> None:
    assert (
        get_compatibility_error(list[int], 42) == "Cannot assign Literal[42] to list\n"
    )
    assert get_compatibility_error(list[int], []) is None
    assert (
        get_compatibility_error(list[int], ["x"])
        == "In element 0\n  Cannot assign Literal['x'] to int\n"
    )
    assert get_compatibility_error(list[int], [1, 2, 3]) is None
