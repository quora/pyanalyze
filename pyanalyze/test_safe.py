from typing import Iterable

from .safe import is_iterable


def test_is_iterable() -> None:
    def gen() -> Iterable[None]:
        yield

    class NoSpecialMethods(object):
        pass

    class HasIter(object):
        def __iter__(self) -> Iterable[int]:
            yield 1

    class HasGetItemAndLen(object):
        def __getitem__(self, i: int) -> int:
            return i ** 2

        def __len__(self) -> int:
            return 1 << 15

    class HasGetItem(object):
        def __getitem__(self, i: int) -> int:
            raise KeyError("tricked you, I am not iterable")

    class HasLen(object):
        def __len__(self) -> int:
            return -1

    assert is_iterable("")
    assert is_iterable([])
    assert is_iterable(range(1))
    assert is_iterable(gen())
    assert is_iterable({})
    assert is_iterable({}.keys())
    assert not is_iterable(42)
    assert not is_iterable(None)
    assert not is_iterable(False)
    assert not is_iterable(str)
    assert not is_iterable(NoSpecialMethods())
    assert is_iterable(HasIter())
    assert is_iterable(HasGetItemAndLen())
    assert not is_iterable(HasGetItem())
    assert not is_iterable(HasLen())
