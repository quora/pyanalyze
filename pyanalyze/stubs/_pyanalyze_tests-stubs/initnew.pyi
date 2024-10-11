from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar, overload

_T = TypeVar("_T")

class simple:
    def __init__(self, x: int) -> None: ...

class my_enumerate(Iterator[tuple[int, _T]], Generic[_T]):
    def __init__(self, iterable: Iterable[_T], start: int = ...) -> None: ...

class overloadinit(Generic[_T]):
    @overload
    def __init__(self, a: int, b: str, c: _T) -> None: ...
    @overload
    def __init__(self, a: str, b: int, c: _T) -> None: ...

class simplenew:
    def __new__(cls, x: int) -> simplenew: ...

class overloadnew(Generic[_T]):
    @overload
    def __new__(cls, a: int, b: str, c: _T) -> overloadnew[_T]: ...
    @overload
    def __new__(cls, a: str, b: int, c: _T) -> overloadnew[_T]: ...
