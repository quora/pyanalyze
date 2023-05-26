from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

def many_defaults(
    a: T = {"a": 1}, b: U = [1, ()], c: V = (1, 2), d: W = {1, 2}
) -> tuple[T, U, V, W]: ...
