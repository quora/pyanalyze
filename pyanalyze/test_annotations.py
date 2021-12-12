# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import skip_before, assert_passes, assert_fails
from .implementation import assert_is_value
from .error_code import ErrorCode
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    KnownValue,
    MultiValuedValue,
    NewTypeValue,
    SequenceIncompleteValue,
    TypeVarValue,
    TypedDictValue,
    TypedValue,
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

    @assert_passes()
    def test_supports_int(self):
        from typing import SupportsInt

        def capybara(z: SupportsInt) -> None:
            assert_is_value(z, TypedValue(SupportsInt))

        def mara():
            capybara(1.0)
            capybara([])  # E: incompatible_argument

    @assert_passes()
    def test_supports_int_accepted(self):
        from typing import SupportsInt

        def capybara(z: SupportsInt) -> None:
            print(z)  # just test that this doesn't get rejected

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
            assert_is_value(capybara(), AnyValue(AnySource.inference))

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

    @assert_passes()
    def test_incompatible_return_value(self):
        def capybara() -> int:
            return "not an int"  # E: incompatible_return_value

    @assert_passes()
    def test_incompatible_return_value_none(self):
        def capybara(x: bool) -> int:
            if not x:
                return  # E: incompatible_return_value
            return 42

    @assert_passes()
    def test_generator(self):
        from typing import Generator

        def capybara(x: bool) -> Generator[int, None, None]:
            if not x:
                return
            yield 42

    @assert_passes()
    def test_incompatible_return_value_pass(self):
        def f() -> int:  # E: missing_return
            pass

    @assert_passes()
    def test_allow_pass_in_abstractmethod(self):
        from abc import abstractmethod

        class X:
            @abstractmethod
            def f(self) -> int:
                pass

    @assert_passes()
    def test_no_return_none(self):
        def f() -> None:
            # TODO this should really be unannotated. The Any comes from _visit_function_body
            # but I'm not sure why we use that one.
            assert_is_value(g(), AnyValue(AnySource.inference))
            return g()  # E: incompatible_return_value

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
            assert_is_value(f(), AnyValue(AnySource.explicit))
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
            assert_is_value(y, AnyValue(AnySource.unannotated))
            assert_is_value(x, TypedValue(int))

    @assert_fails(ErrorCode.incompatible_assignment)
    def test_incompatible_annassign(self):
        def capybara(y: str):
            x: int = y

    @assert_passes()
    def test_typing_tuples(self):
        from typing import Tuple, Union

        def capybara(
            x: Tuple[int, ...],
            y: Tuple[int],
            z: Tuple[str, int],
            omega: Union[Tuple[str, int], None],
            empty: Tuple[()],
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
            assert_is_value(empty, SequenceIncompleteValue(tuple, []))

    @assert_passes()
    def test_stringified_tuples(self):
        from typing import Tuple, Union

        def capybara(
            x: "Tuple[int, ...]",
            y: "Tuple[int]",
            z: "Tuple[str, int]",
            omega: "Union[Tuple[str, int], None]",
            empty: "Tuple[()]",
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
            assert_is_value(empty, SequenceIncompleteValue(tuple, []))

    @skip_before((3, 9))
    @assert_passes()
    def test_builtin_tuples(self):
        from typing import Union

        def capybara(
            x: tuple[int, ...],
            y: tuple[int],
            z: tuple[str, int],
            omega: Union[tuple[str, int], None],
            empty: tuple[()],
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
            assert_is_value(empty, SequenceIncompleteValue(tuple, []))

    @assert_passes()
    def test_invalid_annotation(self):
        def not_an_annotation(x: 1):  # E: invalid_annotation
            pass

        def forward_ref_undefined(x: "NoSuchType"):  # E: undefined_name
            pass

        def forward_ref_bad_attribute(
            x: "collections.defalutdict",  # E: undefined_name
        ):
            pass

        def test_typed_value_annotation() -> dict():  # E: invalid_annotation
            return {}

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

        class Mara:
            x: Final[str] = "x"

        def capybara():
            y: Final = 4
            assert_is_value(x, KnownValue(3))
            assert_is_value(y, KnownValue(4))

            z: Final[int] = 4
            assert_is_value(z, TypedValue(int))

            assert_is_value(Mara().x, TypedValue(str))

    @assert_passes()
    def test_type(self):
        from typing import Type

        def capybara(x: Type[str], y: "Type[int]"):
            assert_is_value(x, SubclassValue(TypedValue(str)))
            assert_is_value(y, SubclassValue(TypedValue(int)))

    @skip_before((3, 9))
    @assert_passes()
    def test_lowercase_type(self):
        def capybara(x: type[str], y: "type[int]"):
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

    @assert_passes()
    def test_stringified_ops(self):
        from typing_extensions import Literal

        def capybara(x: "int | str", y: "Literal[-1]"):
            assert_is_value(x, TypedValue(int) | TypedValue(str))
            assert_is_value(y, KnownValue(-1))

    @assert_passes()
    def test_double_subscript(self):
        from typing import Union, List, Set, TypeVar

        T = TypeVar("T")

        # A bit weird but we hit this kind of case with generic
        # aliases in typeshed.
        def capybara(x: "Union[List[T], Set[T]][int]"):
            assert_is_value(
                x,
                GenericValue(list, [TypedValue(int)])
                | GenericValue(set, [TypedValue(int)]),
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

    @assert_passes()
    def test_classvar(self):
        from typing import ClassVar

        class Capybara:
            x: ClassVar[str]
            Alias: ClassVar = int

            y: Alias

        def caller(c: Capybara):
            assert_is_value(c.x, TypedValue(str))
            assert_is_value(c.y, TypedValue(int))
            assert_is_value(c.Alias, KnownValue(int))


class TestAnnotated(TestNameCheckVisitorBase):
    @assert_passes()
    def test_typing_extensions(self):
        import collections.abc
        from typing_extensions import Annotated
        from typing import Optional, Iterable

        obj = object()

        def capybara(
            x: Annotated[int, "stuff"],
            y: Annotated[int, obj],
            quoted: "Annotated[int, int, str]",
            nested: Annotated[Annotated[int, 1], 2],
            nested_quoted: "Annotated[Annotated[int, 1], 2]",
            in_optional: Optional[Annotated[int, 1]],
            in_iterable: Iterable[Annotated[int, 1]],
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
            assert_is_value(
                in_optional,
                AnnotatedValue(TypedValue(int), [KnownValue(1)]) | KnownValue(None),
            )
            assert_is_value(
                in_iterable,
                GenericValue(
                    collections.abc.Iterable,
                    [AnnotatedValue(TypedValue(int), [KnownValue(1)])],
                ),
            )

    @skip_before((3, 9))
    @assert_passes()
    def test_typing(self):
        import collections.abc
        from typing import Annotated, Iterable, Optional

        obj = object()

        def capybara(
            x: Annotated[int, "stuff"],
            y: Annotated[int, obj],
            quoted: "Annotated[int, int, str]",
            nested: Annotated[Annotated[int, 1], 2],
            nested_quoted: "Annotated[Annotated[int, 1], 2]",
            in_optional: Optional[Annotated[int, 1]],
            in_iterable: Iterable[Annotated[int, 1]],
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
            assert_is_value(
                in_optional,
                AnnotatedValue(TypedValue(int), [KnownValue(1)]) | KnownValue(None),
            )
            assert_is_value(
                in_iterable,
                GenericValue(
                    collections.abc.Iterable,
                    [AnnotatedValue(TypedValue(int), [KnownValue(1)])],
                ),
            )


class TestCallable(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import Callable, Sequence, TypeVar

        T = TypeVar("T")

        def capybara(
            x: Callable[..., int],
            y: Callable[[int], str],
            id_func: Callable[[T], T],
            takes_seq: Callable[[Sequence[T]], T],
            two_args: Callable[[int, str], float],
        ):
            assert_is_value(x(), TypedValue(int))
            assert_is_value(x(arg=3), TypedValue(int))
            assert_is_value(y(1), TypedValue(str))
            assert_is_value(id_func(1), KnownValue(1))
            assert_is_value(takes_seq([int("1")]), TypedValue(int))
            assert_is_value(two_args(1, "x"), TypedValue(float))

    @assert_passes()
    def test_stringified(self):
        from typing import Callable, Sequence, TypeVar

        T = TypeVar("T")

        def capybara(
            x: "Callable[..., int]",
            y: "Callable[[int], str]",
            id_func: "Callable[[T], T]",
            takes_seq: "Callable[[Sequence[T]], T]",
            two_args: "Callable[[int, str], float]",
        ):
            assert_is_value(x(), TypedValue(int))
            assert_is_value(x(arg=3), TypedValue(int))
            assert_is_value(y(1), TypedValue(str))
            assert_is_value(id_func(1), KnownValue(1))
            assert_is_value(takes_seq([int("1")]), TypedValue(int))
            assert_is_value(two_args(1, "x"), TypedValue(float))

    @skip_before((3, 9))
    @assert_passes()
    def test_abc_callable(self):
        from typing import TypeVar
        from collections.abc import Callable, Sequence

        T = TypeVar("T")

        def capybara(
            x: Callable[..., int],
            y: Callable[[int], str],
            id_func: Callable[[T], T],
            takes_seq: Callable[[Sequence[T]], T],
            two_args: Callable[[int, str], float],
        ):
            assert_is_value(x(), TypedValue(int))
            assert_is_value(x(arg=3), TypedValue(int))
            assert_is_value(y(1), TypedValue(str))
            assert_is_value(id_func(1), KnownValue(1))
            assert_is_value(takes_seq([int("1")]), TypedValue(int))
            assert_is_value(two_args(1, "x"), TypedValue(float))

    @assert_passes()
    def test_known_value(self):
        from typing_extensions import Literal
        from typing import Any

        class Capybara:
            def method(self, x: int) -> int:
                return 42

        def f(x: int) -> int:
            return 0

        def g(func: Literal[f]) -> None:
            pass

        def h(x: object) -> bool:
            return True

        def decorator(func: Any) -> Any:
            return func

        def capybara() -> None:
            def nested(x: int) -> int:
                return 2

            @decorator
            def decorated(x: int) -> int:
                return 2

            g(f)
            g(h)
            g(Capybara().method)
            g(nested)
            g(decorated)

    @assert_fails(ErrorCode.incompatible_argument)
    def test_wrong_callable(self):
        from typing import Callable

        def takes_callable(x: Callable[[int], str]) -> None:
            pass

        def wrong_callable(x: str) -> int:
            return 0

        def capybara() -> None:
            takes_callable(wrong_callable)

    @assert_passes()
    def test_known_value_error(self):
        from typing_extensions import Literal

        def f(x: int) -> int:
            return 0

        def g(func: Literal[f]) -> None:
            pass

        def h(x: bool) -> bool:
            return True

        def capybara() -> None:
            g(h)  # E: incompatible_argument

    @assert_passes()
    def test_asynq_callable_incompatible(self):
        from typing import Callable
        from pyanalyze.extensions import AsynqCallable

        def f(x: AsynqCallable[[], int]) -> None:
            pass

        def capybara(func: Callable[[], int]) -> None:
            f(func)  # E: incompatible_argument

    @assert_passes()
    def test_asynq_callable_incompatible_literal(self):
        from pyanalyze.extensions import AsynqCallable

        def f(x: AsynqCallable[[], int]) -> None:
            pass

        def func() -> int:
            return 0

        def capybara() -> None:
            f(func)  # E: incompatible_argument

    @assert_passes()
    def test_asynq_callable(self):
        from asynq import asynq
        from pyanalyze.extensions import AsynqCallable
        from pyanalyze.signature import Signature
        from typing import Optional

        @asynq()
        def func_example(x: int) -> str:
            return ""

        sig = Signature.make([], is_asynq=True, is_ellipsis_args=True)

        @asynq()
        def bare_asynq_callable(fn: AsynqCallable) -> None:
            assert_is_value(fn, CallableValue(sig))
            yield fn.asynq()
            yield fn.asynq("some", "arguments")

        @asynq()
        def caller(
            func: AsynqCallable[[int], str],
            func2: Optional[AsynqCallable[[int], str]] = None,
        ) -> None:
            assert_is_value(func(1), TypedValue(str))
            val = yield func.asynq(1)
            assert_is_value(val, TypedValue(str))
            yield caller.asynq(func_example)
            if func2 is not None:
                yield func2.asynq(1)
            yield bare_asynq_callable.asynq(func_example)

    @assert_passes(settings={ErrorCode.impure_async_call: False})
    def test_amap(self):
        from asynq import asynq
        from pyanalyze.extensions import AsynqCallable
        from typing import TypeVar, List, Iterable

        T = TypeVar("T")
        U = TypeVar("U")

        @asynq()
        def amap(function: AsynqCallable[[T], U], sequence: Iterable[T]) -> List[U]:
            return (yield [function.asynq(elt) for elt in sequence])

        @asynq()
        def mapper(x: int) -> str:
            return ""

        @asynq()
        def caller():
            assert_is_value(amap(mapper, [1]), GenericValue(list, [TypedValue(str)]))
            assert_is_value(
                (yield amap.asynq(mapper, [1])), GenericValue(list, [TypedValue(str)])
            )


class TestTypeVar(TestNameCheckVisitorBase):
    @assert_passes()
    def test_bound(self):
        from typing import TypeVar

        IntT = TypeVar("IntT", bound=int)

        def f(x: IntT) -> IntT:
            assert_is_value(x, TypeVarValue(IntT, bound=TypedValue(int)))
            print(x + 1)
            return x

        def capybara():
            assert_is_value(f(1), KnownValue(1))
            assert_is_value(f(True), KnownValue(True))
            x = f("")  # E: incompatible_argument
            assert_is_value(x, AnyValue(AnySource.error))

    @assert_passes()
    def test_constraint(self):
        from typing import TypeVar, Union

        AnyStr = TypeVar("AnyStr", bytes, str)

        def whatever(x: Union[str, bytes]):
            pass

        def f(x: AnyStr) -> AnyStr:
            print(x.title())
            whatever(x)
            return x

        def capybara(s: str, b: bytes, sb: Union[str, bytes], unannotated):
            assert_is_value(f("x"), TypedValue(str))
            assert_is_value(f(b"x"), TypedValue(bytes))
            assert_is_value(f(s), TypedValue(str))
            assert_is_value(f(b), TypedValue(bytes))
            f(sb)  # E: incompatible_argument
            f(3)  # E: incompatible_argument
            assert_is_value(f(unannotated), AnyValue(AnySource.inference))

    @assert_passes()
    def test_constraint_in_typeshed(self):
        import re

        def capybara():
            assert_is_value(re.escape("x"), TypedValue(str))

    @assert_passes()
    def test_callable_compatibility(self):
        from typing import TypeVar, Callable, Union, Iterable, Any
        from typing_extensions import Protocol

        AnyStr = TypeVar("AnyStr", bytes, str)
        IntT = TypeVar("IntT", bound=int)

        class SupportsIsInteger(Protocol):
            def is_integer(self) -> bool:
                raise NotImplementedError

        SupportsIsIntegerT = TypeVar("SupportsIsIntegerT", bound=SupportsIsInteger)

        def find_int(objs: Iterable[SupportsIsIntegerT]) -> SupportsIsIntegerT:
            for obj in objs:
                if obj.is_integer():
                    return obj
            raise ValueError

        def wants_float_func(f: Callable[[Iterable[float]], float]) -> float:
            return f([1.0, 2.0])

        def want_anystr_func(
            f: Callable[[AnyStr], AnyStr], s: Union[str, bytes]
        ) -> str:
            if isinstance(s, str):
                assert_is_value(f(s), TypedValue(str))
            else:
                assert_is_value(f(s), TypedValue(bytes))
            return ""

        def want_bounded_func(f: Callable[[IntT], IntT], i: int) -> None:
            assert_is_value(f(True), KnownValue(True))
            assert_is_value(f(i), TypedValue(int))

        def want_str_func(f: Callable[[str], str]):
            assert_is_value(f("x"), TypedValue(str))

        def anystr_func(s: AnyStr) -> AnyStr:
            return s

        def int_func(i: IntT) -> IntT:
            return i

        def capybara():
            want_anystr_func(anystr_func, "x")
            want_anystr_func(int_func, "x")  # E: incompatible_argument
            want_bounded_func(int_func, 0)
            want_bounded_func(anystr_func, 1)  # E: incompatible_argument
            want_str_func(anystr_func)
            want_str_func(int_func)  # E: incompatible_argument
            wants_float_func(find_int)
            wants_float_func(int_func)  # E: incompatible_argument

    @assert_passes()
    def test_getitem(self):
        from typing import Any, Dict, TypeVar, Iterable

        T = TypeVar("T", bound=Dict[str, Any])

        def _fetch_credentials(api: T, credential_names: Iterable[str]) -> T:
            api_with_credentials = api.copy()
            for name in credential_names:
                api_with_credentials[name] = str(api[name])
            return api_with_credentials


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

    @assert_passes()
    def test_self(self):
        from pyanalyze.extensions import ParameterTypeGuard
        from typing_extensions import Annotated
        from typing import Union

        class A:
            def is_b(self) -> Annotated[bool, ParameterTypeGuard["self", "B"]]:
                return isinstance(self, B)

        class B(A):
            pass

        class C(A):
            pass

        def capybara(obj: A) -> None:
            assert_is_value(obj, TypedValue(A))
            if obj.is_b():
                assert_is_value(obj, TypedValue(B))
            else:
                assert_is_value(obj, TypedValue(A))

        def narrow_union(union: Union[B, C]) -> None:
            assert_is_value(union, TypedValue(B) | TypedValue(C))
            if union.is_b():
                assert_is_value(union, TypedValue(B))


class TestTypeGuard(TestNameCheckVisitorBase):
    @assert_passes()
    def test_typing_extesions(self):
        from typing_extensions import TypeGuard
        from typing import Union

        def is_int(x: Union[int, str]) -> TypeGuard[int]:
            return x == 42

        def is_quoted_int(x: Union[int, str]) -> "TypeGuard[int]":
            return x == 42

        def capybara(x: Union[int, str]):
            if is_int(x):
                assert_is_value(x, TypedValue(int))
            else:
                assert_is_value(x, MultiValuedValue([TypedValue(int), TypedValue(str)]))

        def pacarana(x: Union[int, str]):
            if is_quoted_int(x):
                assert_is_value(x, TypedValue(int))
            else:
                assert_is_value(x, MultiValuedValue([TypedValue(int), TypedValue(str)]))

    @assert_passes()
    def test(self):
        from pyanalyze.extensions import TypeGuard
        from typing import Union

        def is_int(x: Union[int, str]) -> TypeGuard[int]:
            return x == 42

        def capybara(x: Union[int, str]):
            if is_int(x):
                assert_is_value(x, TypedValue(int))
            else:
                assert_is_value(x, MultiValuedValue([TypedValue(int), TypedValue(str)]))

    @assert_passes()
    def test_method(self) -> None:
        from pyanalyze.extensions import TypeGuard
        from typing import Union

        class Cls:
            def is_int(self, x: Union[int, str]) -> TypeGuard[int]:
                return x == 43

        def capybara(x: Union[int, str]):
            cls = Cls()
            if cls.is_int(x):
                assert_is_value(x, TypedValue(int))
                assert_is_value(cls, TypedValue(Cls))
            else:
                assert_is_value(x, MultiValuedValue([TypedValue(int), TypedValue(str)]))
                assert_is_value(cls, TypedValue(Cls))


class TestCustomCheck(TestNameCheckVisitorBase):
    @assert_passes()
    def test_literal_only(self) -> None:
        from pyanalyze.extensions import LiteralOnly
        from typing_extensions import Annotated

        def capybara(x: Annotated[str, LiteralOnly()]) -> str:
            return x

        def caller(x: str) -> None:
            capybara("x")
            capybara(x)  # E: incompatible_argument
            capybara(str(1))  # E: incompatible_argument
            capybara("x" if x else "y")
            capybara("x" if x else x)  # E: incompatible_argument

    @assert_passes()
    def test_no_any(self) -> None:
        from pyanalyze.extensions import NoAny
        from typing_extensions import Annotated
        from typing import List

        def shallow(x: Annotated[List[int], NoAny()]) -> None:
            pass

        def deep(x: Annotated[List[int], NoAny(deep=True)]) -> None:
            pass

        def none_at_all(
            x: Annotated[List[int], NoAny(deep=True, allowed_sources=frozenset())]
        ) -> None:
            pass

        def capybara(unannotated) -> None:
            shallow(unannotated)  # E: incompatible_argument
            shallow([1])
            shallow([int(unannotated)])
            shallow([unannotated])
            deep(unannotated)  # E: incompatible_argument
            deep([1])
            deep([int(unannotated)])
            deep([unannotated])  # E: incompatible_argument
            none_at_all(unannotated)  # E: incompatible_argument
            none_at_all([1])
            none_at_all([int(unannotated)])
            none_at_all([unannotated])  # E: incompatible_argument

            lst = []
            for x in lst:
                assert_is_value(x, AnyValue(AnySource.unreachable))
                shallow(x)
                deep(x)
                none_at_all(x)  # E: incompatible_argument

    @assert_passes()
    def test_not_none(self) -> None:
        from dataclasses import dataclass
        from pyanalyze.extensions import CustomCheck
        from pyanalyze.value import (
            flatten_values,
            CanAssign,
            CanAssignError,
            CanAssignContext,
            KnownValue,
            Value,
        )
        from typing_extensions import Annotated
        from typing import Any, Optional

        @dataclass(frozen=True)
        class IsNot(CustomCheck):
            obj: object

            def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
                for subval in flatten_values(value):
                    if isinstance(subval, KnownValue):
                        if subval.val is self.obj:
                            return CanAssignError(f"Value may not be {self.obj!r}")
                return {}

        def capybara(x: Annotated[Any, IsNot(None)]) -> None:
            pass

        def caller(x: Optional[str]) -> None:
            capybara("x")
            capybara(None)  # E: incompatible_argument
            capybara(x)  # E: incompatible_argument
            capybara("x" if x else None)  # E: incompatible_argument

    @assert_passes()
    def test_greater_than(self) -> None:
        from dataclasses import dataclass
        from pyanalyze.extensions import CustomCheck
        from pyanalyze.value import (
            flatten_values,
            CanAssign,
            CanAssignError,
            CanAssignContext,
            KnownValue,
            TypeVarMap,
            TypeVarValue,
            Value,
        )
        from typing_extensions import Annotated, TypeGuard
        from typing import Iterable, TypeVar, Union

        @dataclass(frozen=True)
        class GreaterThan(CustomCheck):
            # Must be quoted in 3.6 because otherwise the runtime explodes.
            value: Union[int, "TypeVar"]

            def _can_assign_inner(self, value: Value) -> CanAssign:
                if isinstance(value, KnownValue):
                    if not isinstance(value.val, int):
                        return CanAssignError(f"Value {value.val!r} is not an int")
                    if value.val <= self.value:
                        return CanAssignError(
                            f"Value {value.val!r} is not greater than {self.value}"
                        )
                    return {}
                elif isinstance(value, AnyValue):
                    return {}
                else:
                    return CanAssignError(f"Size of {value} is not known")

            def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
                if isinstance(self.value, TypeVar):
                    return {}
                for subval in flatten_values(value, unwrap_annotated=False):
                    if isinstance(subval, AnnotatedValue):
                        can_assign = self._can_assign_inner(subval.value)
                        if not isinstance(can_assign, CanAssignError):
                            return can_assign
                        gts = list(subval.get_custom_check_of_type(GreaterThan))
                        if not gts:
                            return CanAssignError(f"Size of {value} is not known")
                        if not any(
                            check.value >= self.value
                            for check in gts
                            if isinstance(check.value, int)
                        ):
                            return CanAssignError(f"{subval} is too small")
                    else:
                        can_assign = self._can_assign_inner(subval)
                        if isinstance(can_assign, CanAssignError):
                            return can_assign
                return {}

            def walk_values(self) -> Iterable[Value]:
                if isinstance(self.value, TypeVar):
                    yield TypeVarValue(self.value)

            def substitute_typevars(self, typevars: TypeVarMap) -> "GreaterThan":
                if isinstance(self.value, TypeVar) and self.value in typevars:
                    value = typevars[self.value]
                    if isinstance(value, KnownValue) and isinstance(value.val, int):
                        return GreaterThan(value.val)
                return self

        def capybara(x: Annotated[int, GreaterThan(2)]) -> None:
            pass

        IntT = TypeVar("IntT", bound=int)

        def is_greater_than(
            x: int, limit: IntT
        ) -> TypeGuard[Annotated[int, GreaterThan(IntT)]]:
            return x > limit

        def caller(x: int) -> None:
            capybara(x)  # E: incompatible_argument
            if is_greater_than(x, 2):
                capybara(x)  # ok
            capybara(3)  # ok
            capybara(2)  # E: incompatible_argument


class TestExternalType(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        import os
        from pyanalyze.extensions import ExternalType
        from typing_extensions import Annotated
        from typing import Union

        def capybara(
            x: ExternalType["builtins.str"],
            y: ExternalType["os.stat_result"],
            z: Annotated[ExternalType["builtins.str"], 1] = "z",
            omega: Union[
                ExternalType["builtins.str"], ExternalType["builtins.int"]
            ] = 1,
        ) -> None:
            assert_is_value(x, TypedValue(str))
            assert_is_value(y, TypedValue(os.stat_result))
            assert_is_value(z, AnnotatedValue(TypedValue(str), [KnownValue(1)]))
            assert_is_value(omega, TypedValue(str) | TypedValue(int))

        def user():
            sr = os.stat_result((1,) * 10)
            capybara("x", 1)  # E: incompatible_argument
            capybara(1, sr)  # E: incompatible_argument
            capybara("x", sr)


class TestRequired(TestNameCheckVisitorBase):
    @assert_passes()
    def test_typing_extensions(self):
        from typing_extensions import NotRequired, Required, TypedDict

        class RNR(TypedDict):
            a: int
            b: Required[str]
            c: NotRequired[float]

        def take_rnr(td: RNR) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": (True, TypedValue(int)),
                        "b": (True, TypedValue(str)),
                        "c": (False, TypedValue(float)),
                    }
                ),
            )

        class NotTotal(TypedDict, total=False):
            a: int
            b: Required[str]
            c: NotRequired[float]

        def take_not_total(td: NotTotal) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": (False, TypedValue(int)),
                        "b": (True, TypedValue(str)),
                        "c": (False, TypedValue(float)),
                    }
                ),
            )

        class Stringify(TypedDict):
            a: "int"
            b: "Required[str]"
            c: "NotRequired[float]"

        def take_stringify(td: Stringify) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": (True, TypedValue(int)),
                        "b": (True, TypedValue(str)),
                        "c": (False, TypedValue(float)),
                    }
                ),
            )

    @assert_passes()
    def test_typeddict_from_call(self):
        from typing import Optional, Any
        from typing_extensions import NotRequired, Required, TypedDict

        class Stringify(TypedDict):
            a: "int"
            b: "Required[str]"
            c: "NotRequired[float]"

        def make_td() -> Any:
            return Stringify

        def return_optional() -> Optional[Stringify]:
            return None

        def return_call() -> Optional[make_td()]:
            return None

        def capybara() -> None:
            assert_is_value(
                return_optional(),
                KnownValue(None)
                | TypedDictValue(
                    {
                        "a": (True, TypedValue(int)),
                        "b": (True, TypedValue(str)),
                        "c": (False, TypedValue(float)),
                    }
                ),
            )
            assert_is_value(
                return_call(),
                KnownValue(None)
                | TypedDictValue(
                    {
                        "a": (True, TypedValue(int)),
                        "b": (True, TypedValue(str)),
                        "c": (False, TypedValue(float)),
                    }
                ),
            )

    @skip_before((3, 8))
    @assert_passes()
    def test_typing(self):
        from typing_extensions import NotRequired, Required
        from typing import TypedDict

        class RNR(TypedDict):
            a: int
            b: Required[str]
            c: NotRequired[float]

        def take_rnr(td: RNR) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": (True, TypedValue(int)),
                        "b": (True, TypedValue(str)),
                        "c": (False, TypedValue(float)),
                    }
                ),
            )

        class NotTotal(TypedDict, total=False):
            a: int
            b: Required[str]
            c: NotRequired[float]

        def take_not_total(td: NotTotal) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": (False, TypedValue(int)),
                        "b": (True, TypedValue(str)),
                        "c": (False, TypedValue(float)),
                    }
                ),
            )

        class Stringify(TypedDict):
            a: "int"
            b: "Required[str]"
            c: "NotRequired[float]"

        def take_stringify(td: Stringify) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": (True, TypedValue(int)),
                        "b": (True, TypedValue(str)),
                        "c": (False, TypedValue(float)),
                    }
                ),
            )

    @assert_passes()
    def test_unsupported_location(self):
        from typing_extensions import NotRequired, Required

        def f(x: Required[int]) -> None:  # E: invalid_annotation
            pass

        def g() -> Required[int]:  # E: invalid_annotation
            return 3

        class Capybara:
            x: Required[int]  # E: invalid_annotation
            y: NotRequired[int]  # E: invalid_annotation
