from typing import overload

from typing_extensions import Literal

@overload
def func(x: int, y: Literal[1] = ..., z: str = ...) -> int: ...
@overload
def func(x: int, y: int, z: str = ...) -> str: ...
@overload
def func(x: str, y: int = ..., z: str = ...) -> float: ...
