from typing import TypeVar

# Just testing that the presence of a default doesn't
# completely break type checking.
_T = TypeVar("_T", default=None)

def f(x: _T) -> _T: ...
