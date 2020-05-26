# static analysis: ignore
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import skip_before
from .error_code import ErrorCode


class TestAnnotations(TestNameCheckVisitorBase):
    @skip_before((3, 5))
    def test_union(self):
        self.assert_passes(
            """
import re
from typing import Union, Optional, List, Set, Dict, Match, Pattern

_Pattern = type(re.compile("a"))
_Match = type(re.match("a", "a"))

def capybara() -> Union[int, str]:
    return 0

def kerodon() -> Optional[int]:
    return None

def complex() -> Union[List[str], Set[int], Dict[float, List[str]], int]:
    return []

def check() -> None:
    assert_is_value(capybara(), MultiValuedValue([TypedValue(int), TypedValue(str)]))
    assert_is_value(kerodon(), MultiValuedValue([TypedValue(int), KnownValue(None)]))
    assert_is_value(
        complex(),
        MultiValuedValue(
            [
                GenericValue(list, [TypedValue(str)]),
                GenericValue(set, [TypedValue(int)]),
                GenericValue(
                    dict, [TypedValue(float), GenericValue(list, [TypedValue(str)])]
                ),
                TypedValue(int),
            ]
        ),
    )

def rgx(m: Match[str], p: Pattern[bytes]) -> None:
    assert_is_value(p, GenericValue(_Pattern, [TypedValue(bytes)]))
    assert_is_value(m, GenericValue(_Match, [TypedValue(str)]))
"""
        )

    @skip_before((3, 5))
    def test_generic(self):
        self.assert_passes(
            """
from typing import List, SupportsInt

def capybara(x: List[int], y: List, z: SupportsInt) -> None:
    assert_is_value(x, GenericValue(list, [TypedValue(int)]))
    assert_is_value(y, TypedValue(list))
    assert_is_value(z, TypedValue(SupportsInt))
"""
        )

    @skip_before((3, 5))
    def test_self_type(self):
        self.assert_passes(
            """
class Capybara:
    def f(self: int) -> None:
        assert_is_value(self, TypedValue(int))

    def g(self) -> None:
        assert_is_value(self, TypedValue(Capybara))
"""
        )

    @skip_before((3, 5))
    def test_newtype(self):
        self.assert_passes(
            """
from typing import NewType, Tuple

X = NewType("X", int)
Y = NewType("Y", Tuple[str, ...])

def capybara(x: X, y: Y) -> None:
    assert_is_value(x, NewTypeValue(X))
    print(y)  # just asserting that this doesn't cause errors
"""
        )

    @skip_before((3, 5))
    def test_literal(self):
        self.assert_passes(
            """
from typing_extensions import Literal

def capybara(x: Literal[True], y: Literal[True, False]) -> None:
    assert_is_value(x, KnownValue(True))
    assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))
"""
        )

    @skip_before((3, 5))
    def test_contextmanager(self):
        self.assert_passes(
            """
from contextlib import contextmanager
from typing import Iterator

@contextmanager
def capybara() -> Iterator[int]:
    yield 3

def kerodon():
    # Ideally should be ContextManager[int], but at least
    # it should not be Iterator[int], which is what pyanalyze
    # used to infer.
    assert_is_value(capybara(), UNRESOLVED_VALUE)
"""
        )

    @skip_before((3, 0))
    def test_none_annotations(self):
        self.assert_passes(
            """
def mara() -> None:
    pass

class Capybara:
    def __init__(self) -> None:
        pass

def check() -> None:
    # Make sure we don't infer None if __init__ is annotated
    # as returning None.
    assert_is_value(Capybara(), TypedValue(Capybara))
    assert_is_value(mara(), KnownValue(None))
"""
        )

    @skip_before((3, 0))
    def test_annotations(self):
        self.assert_passes(
            """
def caviidae() -> None:
    x = int
    # tests that annotations in a nested functions are not evaluated in a context where they don't exist
    def capybara(a: x, *b: x, c: x, d: x=3, **kwargs: x):
        pass
    assert_is_value(capybara, KnownValue(capybara))
"""
        )
        self.assert_passes(
            """
class Caviidae:
    class Capybara:
        pass

    def eat(self, x: Capybara):
        assert_is_value(self, TypedValue(Caviidae))

    @staticmethod
    def static(x: "Caviidae"):
        assert_is_value(x, TypedValue(Caviidae))
"""
        )
        self.assert_fails(
            ErrorCode.incompatible_argument,
            """
def capybara(x: int) -> None:
    pass

def kerodon():
    capybara("not an int")
""",
        )

    @skip_before((3, 0))
    def test_incompatible_return_value(self):
        self.assert_fails(
            ErrorCode.incompatible_return_value,
            """
def capybara() -> int:
    return "not an int"
""",
        )
        self.assert_fails(
            ErrorCode.incompatible_return_value,
            """
def capybara(x: bool) -> int:
    if not x:
        return
    return 42
""",
        )
        self.assert_passes(
            """
from typing import Generator

def capybara(x: bool) -> Generator[int, None, None]:
    if not x:
        return
    yield 42
"""
        )
        self.assert_fails(
            ErrorCode.incompatible_return_value,
            """
def f() -> int:
    pass
""",
        )
        self.assert_passes(
            """
from abc import abstractmethod

class X:
    @abstractmethod
    def f(self) -> int:
        pass
""",
        )
        self.assert_fails(
            ErrorCode.incompatible_return_value,
            """
def f() -> None:
    assert_is_value(g(), UNRESOLVED_VALUE)
    return g()

def g():
    pass
""",
        )

    @skip_before((3, 0))
    def test_incompatible_default(self):
        self.assert_fails(
            ErrorCode.incompatible_default,
            """
def capybara(x: int = None) -> None:
    pass
""",
        )

    @skip_before((3, 0))
    def test_property(self):
        self.assert_passes(
            """
class Capybara:
    def __init__(self, x):
        self.x = x

    @property
    def f(self) -> int:
        return self.x

    def get_g(self) -> int:
        return self.x * 2

    g = property(get_g)


def user(c: Capybara) -> None:
    assert_is_value(c.f, TypedValue(int))
    assert_is_value(c.get_g(), TypedValue(int))
    assert_is_value(c.g, TypedValue(int))
"""
        )

    @skip_before((3, 0))
    def test_annotations_override_return(self):
        self.assert_passes(
            """
from typing import Any

def f() -> Any:
    return 0

def g():
    return 0

def capybara():
    assert_is_value(f(), UNRESOLVED_VALUE)
    assert_is_value(g(), KnownValue(0))
"""
        )

    @skip_before((3, 0))
    def test_cached_classmethod(self):
        # just test that this doesn't crash
        self.assert_passes(
            """
from functools import lru_cache

class Capybara:
    @classmethod
    @lru_cache()
    def f(cls) -> int:
        return 3
"""
        )

    @skip_before((3, 6))
    def test_annassign(self):
        self.assert_passes(
            """
def capybara(y):
    x: int = y
    assert_is_value(y, UNRESOLVED_VALUE)
    assert_is_value(x, TypedValue(int))
"""
        )
        self.assert_fails(
            ErrorCode.incompatible_assignment,
            """
def capybara(y: str):
    x: int = y
""",
        )

    @skip_before((3, 5))
    def test_tuples(self):
        self.assert_passes(
            """
from typing import Tuple, Union

def capybara(x: Tuple[int, ...], y: Tuple[int], z: Tuple[str, int], omega: Union[Tuple[str, int], None]) -> None:
    assert_is_value(x, GenericValue(tuple, [TypedValue(int)]))
    assert_is_value(y, SequenceIncompleteValue(tuple, [TypedValue(int)]))
    assert_is_value(z, SequenceIncompleteValue(tuple, [TypedValue(str), TypedValue(int)]))
    assert_is_value(omega, MultiValuedValue([
        SequenceIncompleteValue(tuple, [TypedValue(str), TypedValue(int)]),
        KnownValue(None),
    ]))
"""
        )

    @skip_before((3, 0))
    def test_invalid_annotation(self):
        self.assert_fails(
            ErrorCode.invalid_annotation,
            """
def f(x: 1):
    pass
""",
        )

    @skip_before((3, 0))
    def test_forward_ref(self):
        self.assert_fails(
            ErrorCode.undefined_name,
            """
def f(x: "NoSuchType"):
    pass
""",
        )
        self.assert_passes(
            """
import typing
from typing import Optional

def capybara(x: "X", y: "Optional[X]", z: "typing.Optional[X]"):
    assert_is_value(x, TypedValue(X))
    assert_is_value(y, MultiValuedValue([KnownValue(None), TypedValue(X)]))
    assert_is_value(z, MultiValuedValue([KnownValue(None), TypedValue(X)]))

class X:
    pass
"""
        )
        self.assert_passes(
            """
from typing import List

def capybara(x: "List[int]") -> "List[str]":
    assert_is_value(x, GenericValue(list, [TypedValue(int)]))
    assert_is_value(capybara(x), GenericValue(list, [TypedValue(str)]))
    return []
"""
        )
        self.assert_fails(
            ErrorCode.incompatible_return_value,
            """
def f() -> "int":
    return ""
""",
        )

    @skip_before((3, 0))
    def test_pattern(self):
        self.assert_passes(
            """
from typing import Pattern
import re

_Pattern = type(re.compile(""))

def capybara(x: Pattern[str]):
    assert_is_value(x, GenericValue(_Pattern, [TypedValue(str)]))
"""
        )

    @skip_before((3, 6))
    def test_final(self):
        self.assert_passes(
            """
from typing_extensions import Final

x: Final = 3

def capybara():
    y: Final = 4
    assert_is_value(x, KnownValue(3))
    assert_is_value(y, KnownValue(4))
"""
        )

    @skip_before((3, 6))
    def test_type(self):
        self.assert_passes(
            """
from typing import Type

def capybara(x: Type[str], y: "Type[int]"):
    assert_is_value(x, SubclassValue(str))
    assert_is_value(y, SubclassValue(int))
"""
        )
