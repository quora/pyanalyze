from typing import Callable

from typing_extensions import ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")

def f(x: T) -> T: ...
def g(x: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T: ...
