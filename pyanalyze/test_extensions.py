from qcore.asserts import AssertRaises, assert_eq
import typing_inspect
from typing import List, Optional, Union, TypeVar

from .extensions import AsynqCallable

T = TypeVar("T")
U = TypeVar("U")


def test_asynq_callable() -> None:
    AC = AsynqCallable[[int], str]
    assert_eq((AC, type(None)), typing_inspect.get_args(Optional[AC]))
    assert_eq((int, AC), typing_inspect.get_args(Union[int, AC]))

    GAC = AsynqCallable[[T], str]
    assert_eq(AC, GAC[int])

    assert_eq(
        AsynqCallable[[List[int]], List[str]],
        AsynqCallable[[List[T]], List[U]][int, str],
    )

    with AssertRaises(TypeError):
        # Unfortunately this doesn't work because typing doesn't know how to
        # get TypeVars out of an AsynqCallable instances. Solving this is hard
        # because Callable is special-cased at various places in typing.py.
        assert_eq(List[AsynqCallable[[str], int]], List[AsynqCallable[[T], int]][str])
