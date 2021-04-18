# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import skip_before, assert_passes, assert_fails
from .implementation import assert_is_value
from .error_code import ErrorCode
from .value import (
    AnnotatedValue,
    KnownValue,
    MultiValuedValue,
    NewTypeValue,
    SequenceIncompleteValue,
    TypedValue,
    UNRESOLVED_VALUE,
    SubclassValue,
    GenericValue,
)


class TestAnnotations(TestNameCheckVisitorBase):
    @assert_passes()
    def test_union(self):
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

        def union_in_subscript() -> List[Union[str, int]]:
            return []

        def check() -> None:
            assert_is_value(
                capybara(), MultiValuedValue([TypedValue(int), TypedValue(str)])
            )
            assert_is_value(
                kerodon(), MultiValuedValue([TypedValue(int), KnownValue(None)])
            )
            assert_is_value(
                complex(),
                MultiValuedValue(
                    [
                        GenericValue(list, [TypedValue(str)]),
                        GenericValue(set, [TypedValue(int)]),
                        GenericValue(
                            dict,
                            [TypedValue(float), GenericValue(list, [TypedValue(str)])],
                        ),
                        TypedValue(int),
                    ]
                ),
            )
            assert_is_value(
                union_in_subscript(),
                GenericValue(
                    list, [MultiValuedValue([TypedValue(str), TypedValue(int)])]
                ),
            )

        def rgx(m: Match[str], p: Pattern[bytes]) -> None:
            assert_is_value(p, GenericValue(_Pattern, [TypedValue(bytes)]))
            assert_is_value(m, GenericValue(_Match, [TypedValue(str)]))

    @assert_passes()
    def test_generic(self):
        from typing import List

        def capybara(x: List[int], y: List) -> None:
            assert_is_value(x, GenericValue(list, [TypedValue(int)]))
            assert_is_value(y, TypedValue(list))

    # on 3.6 and 3.7 SupportsInt becomes UNRESOLVED_VALUE because it's not
    # runtime checkable.
    @skip_before((3, 8))
    @assert_passes()
    def test_supports_int(self):
        from typing import SupportsInt

        def capybara(z: SupportsInt) -> None:
            assert_is_value(z, TypedValue(SupportsInt))

    @assert_passes()
    def test_self_type(self):
        class Capybara:
            def f(self: int) -> None:
                assert_is_value(self, TypedValue(int))

            def g(self) -> None:
                assert_is_value(self, TypedValue(Capybara))

    @assert_passes()
    def test_newtype(self):
        from typing import NewType, Tuple

        X = NewType("X", int)
        Y = NewType("Y", Tuple[str, ...])

        def capybara(x: X, y: Y) -> None:
            assert_is_value(x, NewTypeValue(X))
            print(y)  # just asserting that this doesn't cause errors

    @assert_passes()
    def test_literal(self):
        from typing_extensions import Literal

        def capybara(x: Literal[True], y: Literal[True, False]) -> None:
            assert_is_value(x, KnownValue(True))
            assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))

    @assert_passes()
    def test_contextmanager(self):
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

    @assert_passes()
    def test_none_annotations(self):
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

    @assert_passes()
    def test_annotations_function(self):
        def caviidae() -> None:
            x = int
            # tests that annotations in a nested functions are not evaluated in a context where they don't exist
            def capybara(a: x, *b: x, c: x, d: x = 3, **kwargs: x):
                pass

            assert_is_value(capybara, KnownValue(capybara))

    @assert_passes()
    def annotations_class(self):
        class Caviidae:
            class Capybara:
                pass

            def eat(self, x: Capybara):
                assert_is_value(self, TypedValue(Caviidae))

            @staticmethod
            def static(x: "Caviidae"):
                assert_is_value(x, TypedValue(Caviidae))

    @assert_fails(ErrorCode.incompatible_argument)
    def test_incompatible_annotations(self):
        def capybara(x: int) -> None:
            pass

        def kerodon():
            capybara("not an int")

    @assert_fails(ErrorCode.incompatible_return_value)
    def test_incompatible_return_value(self):
        def capybara() -> int:
            return "not an int"

    @assert_fails(ErrorCode.incompatible_return_value)
    def test_incompatible_return_value_none(self):
        def capybara(x: bool) -> int:
            if not x:
                return
            return 42

    @assert_passes()
    def test_generator(self):
        from typing import Generator

        def capybara(x: bool) -> Generator[int, None, None]:
            if not x:
                return
            yield 42

    @assert_fails(ErrorCode.incompatible_return_value)
    def test_incompatible_return_value_pass(self):
        def f() -> int:
            pass

    @assert_passes()
    def test_allow_pass_in_abstractmethod(self):
        from abc import abstractmethod

        class X:
            @abstractmethod
            def f(self) -> int:
                pass

    @assert_fails(ErrorCode.incompatible_return_value)
    def test_no_return_none(self):
        def f() -> None:
            assert_is_value(g(), UNRESOLVED_VALUE)
            return g()

        def g():
            pass

    @assert_fails(ErrorCode.incompatible_default)
    def test_incompatible_default(self):
        def capybara(x: int = None) -> None:
            pass

    @assert_passes()
    def test_property(self):
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

    @assert_passes()
    def test_annotations_override_return(self):
        from typing import Any

        def f() -> Any:
            return 0

        def g():
            return 0

        def capybara():
            assert_is_value(f(), UNRESOLVED_VALUE)
            assert_is_value(g(), KnownValue(0))

    @assert_passes()
    def test_cached_classmethod(self):
        # just test that this doesn't crash
        from functools import lru_cache

        class Capybara:
            @classmethod
            @lru_cache()
            def f(cls) -> int:
                return 3

    @assert_passes()
    def test_annassign(self):
        def capybara(y):
            x: int = y
            assert_is_value(y, UNRESOLVED_VALUE)
            assert_is_value(x, TypedValue(int))

    @assert_fails(ErrorCode.incompatible_assignment)
    def test_incompatible_annassign(self):
        def capybara(y: str):
            x: int = y

    @assert_passes()
    def test_tuples(self):
        from typing import Tuple, Union

        def capybara(
            x: Tuple[int, ...],
            y: Tuple[int],
            z: Tuple[str, int],
            omega: Union[Tuple[str, int], None],
        ) -> None:
            assert_is_value(x, GenericValue(tuple, [TypedValue(int)]))
            assert_is_value(y, SequenceIncompleteValue(tuple, [TypedValue(int)]))
            assert_is_value(
                z, SequenceIncompleteValue(tuple, [TypedValue(str), TypedValue(int)])
            )
            assert_is_value(
                omega,
                MultiValuedValue(
                    [
                        SequenceIncompleteValue(
                            tuple, [TypedValue(str), TypedValue(int)]
                        ),
                        KnownValue(None),
                    ]
                ),
            )

    @assert_fails(ErrorCode.invalid_annotation)
    def test_invalid_annotation(self):
        def f(x: 1):
            pass

    @assert_fails(ErrorCode.undefined_name)
    def test_forward_ref_undefined(self):
        def f(x: "NoSuchType"):
            pass

    @assert_passes()
    def test_forward_ref_optional(self):
        import typing
        from typing import Optional

        def capybara(x: "X", y: "Optional[X]", z: "typing.Optional[X]"):
            assert_is_value(x, TypedValue(X))
            assert_is_value(y, MultiValuedValue([KnownValue(None), TypedValue(X)]))
            assert_is_value(z, MultiValuedValue([KnownValue(None), TypedValue(X)]))

        class X:
            pass

    @assert_passes()
    def test_forward_ref_list(self):
        from typing import List

        def capybara(x: "List[int]") -> "List[str]":
            assert_is_value(x, GenericValue(list, [TypedValue(int)]))
            assert_is_value(capybara(x), GenericValue(list, [TypedValue(str)]))
            return []

    @assert_fails(ErrorCode.incompatible_return_value)
    def test_forward_ref_incompatible(self):
        def f() -> "int":
            return ""

    @assert_passes()
    def test_pattern(self):
        from typing import Pattern
        import re

        _Pattern = type(re.compile(""))

        def capybara(x: Pattern[str]):
            assert_is_value(x, GenericValue(_Pattern, [TypedValue(str)]))

    @skip_before((3, 7))
    def test_future_annotations(self):
        self.assert_passes(
            """
from __future__ import annotations
from typing import List

def f(x: int, y: List[str]):
    assert_is_value(x, TypedValue(int))
    assert_is_value(y, GenericValue(list, [TypedValue(str)]))
"""
        )

    @assert_passes()
    def test_final(self):
        from typing_extensions import Final

        x: Final = 3

        def capybara():
            y: Final = 4
            assert_is_value(x, KnownValue(3))
            assert_is_value(y, KnownValue(4))

    @assert_passes()
    def test_type(self):
        from typing import Type

        def capybara(x: Type[str], y: "Type[int]"):
            assert_is_value(x, SubclassValue(TypedValue(str)))
            assert_is_value(y, SubclassValue(TypedValue(int)))

    @skip_before((3, 9))
    @assert_passes()
    def test_generic_alias(self):
        from queue import Queue

        class I:
            ...

        class X:
            def __init__(self):
                self.q: Queue[I] = Queue()

        def f(x: Queue[I]) -> None:
            assert_is_value(x, GenericValue(Queue, [TypedValue(I)]))

        def capybara(x: list[int], y: tuple[int, str], z: tuple[int, ...]) -> None:
            assert_is_value(x, GenericValue(list, [TypedValue(int)]))
            assert_is_value(
                y, SequenceIncompleteValue(tuple, [TypedValue(int), TypedValue(str)])
            )
            assert_is_value(z, GenericValue(tuple, [TypedValue(int)]))

    @skip_before((3, 9))
    def test_pep604(self):
        self.assert_passes(
            """
from __future__ import annotations

def capybara(x: int | None, y: int | str) -> None:
    assert_is_value(x, MultiValuedValue([TypedValue(int), KnownValue(None)]))
    assert_is_value(y, MultiValuedValue([TypedValue(int), TypedValue(str)]))
"""
        )

    @skip_before((3, 8))
    @assert_fails(ErrorCode.incompatible_argument)
    def test_initvar(self):
        from dataclasses import dataclass, InitVar

        @dataclass
        class Capybara:
            x: InitVar[str]

        def f():
            Capybara(x=3)


class TestAnnotated(TestNameCheckVisitorBase):
    @assert_passes()
    def test_typing_extensions(self):
        from typing_extensions import Annotated

        obj = object()

        def capybara(
            x: Annotated[int, "stuff"],
            y: Annotated[int, obj],
            quoted: "Annotated[int, int, str]",
            nested: Annotated[Annotated[int, 1], 2],
            nested_quoted: "Annotated[Annotated[int, 1], 2]",
        ) -> None:
            assert_is_value(x, AnnotatedValue(TypedValue(int), [KnownValue("stuff")]))
            assert_is_value(y, AnnotatedValue(TypedValue(int), [KnownValue(obj)]))
            assert_is_value(
                quoted,
                AnnotatedValue(TypedValue(int), [KnownValue(int), KnownValue(str)]),
            )
            assert_is_value(
                nested, AnnotatedValue(TypedValue(int), [KnownValue(1), KnownValue(2)])
            )
            assert_is_value(
                nested_quoted,
                AnnotatedValue(TypedValue(int), [KnownValue(1), KnownValue(2)]),
            )

    @skip_before((3, 9))
    @assert_passes()
    def test_typing(self):
        from typing import Annotated

        obj = object()

        def capybara(
            x: Annotated[int, "stuff"],
            y: Annotated[int, obj],
            quoted: "Annotated[int, int, str]",
            nested: Annotated[Annotated[int, 1], 2],
            nested_quoted: "Annotated[Annotated[int, 1], 2]",
        ) -> None:
            assert_is_value(x, AnnotatedValue(TypedValue(int), [KnownValue("stuff")]))
            assert_is_value(y, AnnotatedValue(TypedValue(int), [KnownValue(obj)]))
            assert_is_value(
                quoted,
                AnnotatedValue(TypedValue(int), [KnownValue(int), KnownValue(str)]),
            )
            assert_is_value(
                nested, AnnotatedValue(TypedValue(int), [KnownValue(1), KnownValue(2)])
            )
            assert_is_value(
                nested_quoted,
                AnnotatedValue(TypedValue(int), [KnownValue(1), KnownValue(2)]),
            )


class TestParameterTypeGuard(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        from pyanalyze.extensions import ParameterTypeGuard
        from typing_extensions import Annotated

        def is_int(x: object) -> Annotated[bool, ParameterTypeGuard["x", int]]:
            return isinstance(x, int)

        def capybara(x: object) -> None:
            assert_is_value(x, TypedValue(object))
            if is_int(x):
                assert_is_value(x, TypedValue(int))

    @assert_passes()
    def test_generic(self):
        import collections.abc
        from pyanalyze.extensions import ParameterTypeGuard
        from typing_extensions import Annotated
        from typing import TypeVar, Type, Iterable, Union

        T = TypeVar("T")

        def all_of_type(
            elts: Iterable[object], typ: Type[T]
        ) -> Annotated[bool, ParameterTypeGuard["elts", Iterable[T]]]:
            return all(isinstance(elt, typ) for elt in elts)

        def capybara(elts: Iterable[Union[int, str]]) -> None:
            assert_is_value(
                elts,
                GenericValue(
                    collections.abc.Iterable,
                    [MultiValuedValue([TypedValue(int), TypedValue(str)])],
                ),
            )
            if all_of_type(elts, int):
                assert_is_value(
                    elts, GenericValue(collections.abc.Iterable, [TypedValue(int)])
                )
