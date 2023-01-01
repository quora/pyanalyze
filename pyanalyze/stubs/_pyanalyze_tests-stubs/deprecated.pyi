from pyanalyze.extensions import deprecated
from typing import overload

@overload
@deprecated("int support is deprecated")
def deprecated_overload(x: int) -> int: ...
@overload
def deprecated_overload(x: str) -> str: ...
