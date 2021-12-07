from qcore.asserts import AssertRaises
import typing_inspect
from typing import List, Optional, Union, TypeVar
from types import FunctionType

from .extensions import AsynqCallable, get_overloads, overload
from .safe import all_of_type

T = TypeVar("T")
U = TypeVar("U")


def test_asynq_callable() -> None:
    AC = AsynqCallable[[int], str]
    assert (AC, type(None)) == typing_inspect.get_args(Optional[AC])
    assert (int, AC) == typing_inspect.get_args(Union[int, AC])

    GAC = AsynqCallable[[T], str]
    assert AC == GAC[int]

    assert (
        AsynqCallable[[List[int]], List[str]]
        == AsynqCallable[[List[T]], List[U]][int, str]
    )

    with AssertRaises(TypeError):
        # Unfortunately this doesn't work because typing doesn't know how to
        # get TypeVars out of an AsynqCallable instances. Solving this is hard
        # because Callable is special-cased at various places in typing.py.
        assert List[AsynqCallable[[str], int]] == List[AsynqCallable[[T], int]][str]


@overload
def f() -> int:
    raise NotImplementedError


@overload
def f(a: int) -> str:
    raise NotImplementedError


def f(*args: object) -> object:
    raise NotImplementedError


class WithOverloadedMethods:
    @overload
    def f(self) -> int:
        raise NotImplementedError

    @overload
    def f(self, a: int) -> str:
        raise NotImplementedError

    def f(self, *args: object) -> object:
        raise NotImplementedError


def test_overload() -> None:
    overloads = get_overloads("pyanalyze.test_extensions.f")
    assert len(overloads) == 2
    assert all_of_type(overloads, FunctionType)
    assert f not in overloads
    assert overloads[0].__code__.co_argcount == 0
    assert overloads[1].__code__.co_argcount == 1

    method_overloads = get_overloads(
        "pyanalyze.test_extensions.WithOverloadedMethods.f"
    )
    assert len(method_overloads) == 2
    assert all_of_type(method_overloads, FunctionType)
    assert WithOverloadedMethods.f not in overloads
    assert method_overloads[0].__code__.co_argcount == 1
    assert method_overloads[1].__code__.co_argcount == 2
