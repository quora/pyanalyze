# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestTypeIs(TestNameCheckVisitorBase):
    @assert_passes()
    def testTypeIsBasic(self):
        from typing_extensions import TypeIs, assert_type

        class Point:
            pass

        def is_point(a: object) -> TypeIs[Point]:
            return False

        def main(a: object) -> None:
            if is_point(a):
                assert_type(a, Point)
            else:
                assert_type(a, object)

    @assert_passes()
    def testTypeIsTypeArgsNone(self):
        from typing_extensions import TypeIs

        def foo(a: object) -> TypeIs:  # E: invalid_annotation
            return False

    @assert_passes()
    def testTypeIsTypeArgsTooMany(self):
        from typing_extensions import TypeIs

        def foo(a: object) -> "TypeIs[int, int]":  # E: invalid_annotation
            return False

    @assert_passes()
    def testTypeIsTypeArgType(self):
        from typing_extensions import TypeIs

        def foo(a: object) -> "TypeIs[42]":  # E: invalid_annotation
            return False

    @assert_passes()
    def testTypeIsCallArgsNone(self):
        from typing_extensions import TypeIs, assert_type

        class Point:
            pass

        def is_point() -> TypeIs[Point]:  # E: invalid_typeguard
            return False

        def main(a: object) -> None:
            if is_point():
                assert_type(a, object)

    @assert_passes()
    def testTypeIsCallArgsMultiple(self):
        from typing_extensions import TypeIs, assert_type

        class Point:
            pass

        def is_point(a: object, b: object) -> TypeIs[Point]:
            return False

        def main(a: object, b: object) -> None:
            if is_point(a, b):
                assert_type(a, Point)
                assert_type(b, object)

    @assert_passes()
    def testTypeIsWithTypeVar(self):
        from typing import Tuple, Type, TypeVar

        from typing_extensions import TypeIs, assert_type

        T = TypeVar("T")

        def is_tuple_of_type(
            a: Tuple[object, ...], typ: Type[T]
        ) -> TypeIs[Tuple[T, ...]]:
            return False

        def main(a: Tuple[object, ...]):
            if is_tuple_of_type(a, int):
                assert_type(a, Tuple[int, ...])

    @assert_passes()
    def testTypeIsUnionIn(self):
        from typing import Union

        from typing_extensions import TypeIs, assert_type

        def is_foo(a: Union[int, str]) -> TypeIs[str]:
            return False

        def main(a: Union[str, int]) -> None:
            if is_foo(a):
                assert_type(a, str)
            else:
                assert_type(a, int)
            assert_type(a, Union[str, int])

    @assert_passes()
    def testTypeIsUnionOut(self):
        from typing import Union

        from typing_extensions import TypeIs, assert_type

        def is_foo(a: object) -> TypeIs[Union[int, str]]:
            return False

        def main(a: object) -> None:
            if is_foo(a):
                assert_type(a, Union[int, str])

    @assert_passes()
    def testTypeIsNonzeroFloat(self):
        from typing_extensions import TypeIs, assert_type

        def is_nonzero(a: object) -> TypeIs[float]:
            return False

        def main(a: int):
            if is_nonzero(a):
                assert_type(a, int)

    @assert_passes()
    def testTypeIsHigherOrder(self):
        from typing import Callable, Iterable, List, TypeVar

        from typing_extensions import TypeIs, assert_type

        T = TypeVar("T")
        R = TypeVar("R")

        def filter(f: Callable[[T], TypeIs[R]], it: Iterable[T]) -> Iterable[R]:
            return ()

        def is_float(a: object) -> TypeIs[float]:
            return False

        def capybara() -> None:
            a: List[object] = ["a", 0, 0.0]
            b = filter(is_float, a)
            assert_type(b, Iterable[float])

    @assert_passes()
    def testTypeIsMethod(self):
        from typing_extensions import TypeIs, assert_type

        class C:
            def main(self, a: object) -> None:
                if self.is_float(a):
                    assert_type(self, C)
                    assert_type(a, float)

            def is_float(self, a: object) -> TypeIs[float]:
                return False

    @assert_passes()
    def testTypeIsBodyRequiresBool(self):
        from typing_extensions import TypeIs

        def is_float(a: object) -> TypeIs[float]:
            return "not a bool"  # E: incompatible_return_value

    @assert_passes()
    def testTypeIsNarrowToTypedDict(self):
        from typing import Mapping, TypedDict

        from typing_extensions import TypeIs, assert_type

        class User(TypedDict):
            name: str
            id: int

        def is_user(a: Mapping[str, object]) -> TypeIs[User]:
            return isinstance(a.get("name"), str) and isinstance(a.get("id"), int)

        def main(a: Mapping[str, object]) -> None:
            if is_user(a):
                assert_type(a, User)

    @assert_passes()
    def testTypeIsInAssert(self):
        from typing_extensions import TypeIs, assert_type

        def is_float(a: object) -> TypeIs[float]:
            return False

        def main(a: object) -> None:
            assert is_float(a)
            assert_type(a, float)

    @assert_passes()
    def testTypeIsFromAny(self):
        from typing import Any

        from typing_extensions import TypeIs, assert_type

        def is_objfloat(a: object) -> TypeIs[float]:
            return False

        def is_anyfloat(a: Any) -> TypeIs[float]:
            return False

        def objmain(a: object) -> None:
            if is_objfloat(a):
                assert_type(a, float)
            if is_anyfloat(a):
                assert_type(a, float)

        def anymain(a: Any) -> None:
            if is_objfloat(a):
                assert_type(a, float)
            if is_anyfloat(a):
                assert_type(a, float)

    @assert_passes()
    def testTypeIsNegatedAndElse(self):
        from typing import Union

        from typing_extensions import TypeIs, assert_type

        def is_int(a: object) -> TypeIs[int]:
            return False

        def is_str(a: object) -> TypeIs[str]:
            return False

        def intmain(a: Union[int, str]) -> None:
            if not is_int(a):
                assert_type(a, str)
            else:
                assert_type(a, int)

        def strmain(a: Union[int, str]) -> None:
            if is_str(a):
                assert_type(a, str)
            else:
                assert_type(a, int)

    @assert_passes()
    def testTypeIsClassMethod(self):
        from typing_extensions import TypeIs, assert_type

        class C:
            @classmethod
            def is_float(cls, a: object) -> TypeIs[float]:
                return False

            def method(self, a: object) -> None:
                if self.is_float(a):
                    assert_type(a, float)

        def main(a: object) -> None:
            if C.is_float(a):
                assert_type(a, float)

    @assert_passes()
    def testTypeIsRequiresPositionalArgs(self):
        from typing_extensions import TypeIs, assert_type

        def is_float(a: object, b: object = 0) -> TypeIs[float]:
            return False

        def main1(a: object) -> None:
            if is_float(a=a, b=1):
                assert_type(a, float)

            if is_float(b=1, a=a):
                assert_type(a, float)

    @assert_passes()
    def testTypeIsOverload(self):
        from typing import Callable, Iterable, Iterator, List, Optional, TypeVar

        from typing_extensions import TypeIs, assert_type, overload

        T = TypeVar("T")
        R = TypeVar("R")

        @overload
        def filter(f: Callable[[T], TypeIs[R]], it: Iterable[T]) -> Iterator[R]:
            raise NotImplementedError

        @overload
        def filter(f: Callable[[T], bool], it: Iterable[T]) -> Iterator[T]:
            raise NotImplementedError

        def filter(*args):
            pass

        def is_int_typeguard(a: object) -> TypeIs[int]:
            return False

        def is_int_bool(a: object) -> bool:
            return False

        def main(a: List[Optional[int]]) -> None:
            bb = filter(lambda x: x is not None, a)
            assert_type(bb, Iterator[Optional[int]])
            cc = filter(is_int_typeguard, a)
            assert_type(cc, Iterator[int])
            dd = filter(is_int_bool, a)
            assert_type(dd, Iterator[Optional[int]])

    @assert_passes()
    def testTypeIsDecorated(self):
        from typing import TypeVar

        from typing_extensions import TypeIs, assert_type

        T = TypeVar("T")

        def decorator(f: T) -> T:
            return f

        @decorator
        def is_float(a: object) -> TypeIs[float]:
            return False

        def main(a: object) -> None:
            if is_float(a):
                assert_type(a, float)

    @assert_passes()
    def testTypeIsMethodOverride(self):
        from typing_extensions import TypeIs

        class C:
            def is_float(self, a: object) -> TypeIs[float]:
                return False

        class D(C):
            def is_float(self, a: object) -> bool:  # E: incompatible_override
                return False

    @assert_passes()
    def testTypeIsInAnd(self):
        from typing import Any

        from typing_extensions import TypeIs

        def isclass(a: object) -> bool:
            return False

        def ismethod(a: object) -> TypeIs[float]:
            return False

        def isfunction(a: object) -> TypeIs[str]:
            return False

        def isclassmethod(obj: Any) -> bool:
            if (
                ismethod(obj)
                and obj.__self__ is not None  # E: undefined_attribute
                and isclass(obj.__self__)  # E: undefined_attribute
            ):
                return True

            return False

        def coverage(obj: Any) -> bool:
            if not (ismethod(obj) or isfunction(obj)):
                return True
            return False

    @assert_passes()
    def testAssignToTypeIsedVariable1(self):
        from typing_extensions import TypeIs

        class A:
            pass

        class B(A):
            pass

        def guard(a: A) -> TypeIs[B]:
            return False

        def capybara() -> None:
            a = A()
            if not guard(a):
                a = A()
            print(a)

    @assert_passes()
    def testAssignToTypeIsedVariable2(self):
        from typing_extensions import TypeIs

        class A:
            pass

        class B:
            pass

        def guard(a: object) -> TypeIs[B]:
            return False

        def capybara() -> None:
            a = A()
            if not guard(a):
                a = A()
            print(a)

    @assert_passes()
    def testAssignToTypeIsedVariable3(self):
        from typing_extensions import Never, TypeIs, assert_type

        class A:
            pass

        class B:
            pass

        def guard(a: object) -> TypeIs[B]:
            return False

        def capybara() -> None:
            a = A()
            if guard(a):
                assert_type(a, Never)  # TODO A & B
                a = B()
                assert_type(a, B)
                a = A()
                assert_type(a, A)
            assert_type(a, A)

    @assert_passes()
    def testTypeIsNestedRestrictionAny(self):
        from typing import Any, Union

        from typing_extensions import TypeIs, assert_type

        class A: ...

        def f(x: object) -> TypeIs[A]:
            return False

        def g(x: object) -> None: ...

        def test(x: Any) -> None:
            if not (f(x) or x):
                return
            assert_type(x, Union[A, Any])

    @assert_passes()
    def testTypeIsNestedRestrictionUnionOther(self):
        from typing import Union

        from typing_extensions import TypeIs, assert_type

        class A: ...

        class B: ...

        def f(x: object) -> TypeIs[A]:
            return False

        def f2(x: object) -> TypeIs[B]:
            return False

        def test(x: object) -> None:
            if not (f(x) or f2(x)):
                return
            assert_type(x, Union[A, B])

    @assert_passes()
    def testTypeIsComprehensionSubtype(self):
        from typing import List

        from typing_extensions import TypeIs

        class Base: ...

        class Foo(Base): ...

        class Bar(Base): ...

        def is_foo(item: object) -> TypeIs[Foo]:
            return isinstance(item, Foo)

        def is_bar(item: object) -> TypeIs[Bar]:
            return isinstance(item, Bar)

        def foobar(items: List[object]) -> object:
            a: List[Base] = [x for x in items if is_foo(x) or is_bar(x)]
            b: List[Base] = [x for x in items if is_foo(x)]
            c: List[Bar] = [x for x in items if is_foo(x)]  # E: incompatible_assignment
            return (a, b, c)

    @assert_passes()
    def testTypeIsNestedRestrictionUnionIsInstance(self):
        from typing import Any, List

        from typing_extensions import TypeIs, assert_type

        class A: ...

        def f(x: List[Any]) -> TypeIs[List[str]]:
            return False

        def test(x: List[Any]) -> None:
            if not (f(x) or isinstance(x, A)):
                return
            assert_type(x, List[Any])

    @assert_passes()
    def testTypeIsMultipleCondition(self):
        from typing_extensions import Never, TypeIs, assert_type

        class Foo: ...

        class Bar: ...

        def is_foo(item: object) -> TypeIs[Foo]:
            return isinstance(item, Foo)

        def is_bar(item: object) -> TypeIs[Bar]:
            return isinstance(item, Bar)

        def foobar(x: object):
            if not isinstance(x, Foo) or not isinstance(x, Bar):
                return
            assert_type(x, Never)

        def foobar_typeguard(x: object):
            if not is_foo(x) or not is_bar(x):
                return
            assert_type(x, Never)

    @assert_passes()
    def testTypeIsAsFunctionArgAsBoolSubtype(self):
        from typing import Callable

        from typing_extensions import TypeIs

        def accepts_bool(f: Callable[[object], bool]) -> None:
            pass

        def with_bool_typeguard(o: object) -> TypeIs[bool]:
            return False

        def with_str_typeguard(o: object) -> TypeIs[str]:
            return False

        def with_bool(o: object) -> bool:
            return False

        accepts_bool(with_bool_typeguard)
        accepts_bool(with_str_typeguard)
        accepts_bool(with_bool)

    @assert_passes()
    def testTypeIsAsFunctionArg(self):
        from typing import Callable

        from typing_extensions import TypeIs

        def accepts_typeguard(f: Callable[[object], TypeIs[bool]]) -> None:
            pass

        def different_typeguard(f: Callable[[object], TypeIs[str]]) -> None:
            pass

        def with_typeguard(o: object) -> TypeIs[bool]:
            return False

        def with_bool(o: object) -> bool:
            return False

        accepts_typeguard(with_typeguard)
        accepts_typeguard(with_bool)  # E: incompatible_argument

        different_typeguard(with_typeguard)  # E: incompatible_argument
        different_typeguard(with_bool)  # E: incompatible_argument

    @assert_passes()
    def testTypeIsAsGenericFunctionArg(self):
        from typing import Callable, TypeVar

        from typing_extensions import TypeIs

        T = TypeVar("T")

        def accepts_typeguard(f: Callable[[object], TypeIs[T]]) -> None:
            pass

        def with_bool_typeguard(o: object) -> TypeIs[bool]:
            return False

        def with_str_typeguard(o: object) -> TypeIs[str]:
            return False

        def with_bool(o: object) -> bool:
            return False

        accepts_typeguard(with_bool_typeguard)
        accepts_typeguard(with_str_typeguard)
        accepts_typeguard(with_bool)  # E: incompatible_argument

    @assert_passes()
    def testTypeIsAsOverloadedFunctionArg(self):
        # https://github.com/python/mypy/issues/11307
        from typing import Any, Callable, Generic, TypeVar, overload

        from typing_extensions import TypeIs

        _T = TypeVar("_T")

        class filter(Generic[_T]):
            @overload
            def __init__(self, function: Callable[[object], TypeIs[_T]]) -> None:
                pass

            @overload
            def __init__(self, function: Callable[[_T], Any]) -> None:
                pass

            def __init__(self, function):
                pass

        def is_int_typeguard(a: object) -> TypeIs[int]:
            return False

        def returns_bool(a: object) -> bool:
            return False

        def capybara() -> None:
            pass
            # TODO:
            # assert_type(filter(is_int_typeguard), filter[int])
            # assert_type(filter(returns_bool), filter[object])

    @assert_passes()
    def testTypeIsSubtypingVariance(self):
        from typing import Callable

        from typing_extensions import TypeIs

        class A:
            pass

        class B(A):
            pass

        class C(B):
            pass

        def accepts_typeguard(f: Callable[[object], TypeIs[B]]) -> None:
            pass

        def with_typeguard_a(o: object) -> TypeIs[A]:
            return False

        def with_typeguard_b(o: object) -> TypeIs[B]:
            return False

        def with_typeguard_c(o: object) -> TypeIs[C]:
            return False

        accepts_typeguard(with_typeguard_a)  # E: incompatible_argument
        accepts_typeguard(with_typeguard_b)
        accepts_typeguard(with_typeguard_c)  # E: incompatible_argument

    @assert_passes()
    def testTypeIsWithIdentityGeneric(self):
        from typing import TypeVar

        from typing_extensions import TypeIs, assert_type

        _T = TypeVar("_T")

        def identity(val: _T) -> TypeIs[_T]:
            return False

        def func1(name: _T):
            assert_type(name, _T)
            if identity(name):
                pass  # TODO: assert_type(name, _T)

        def func2(name: str):
            assert_type(name, str)
            if identity(name):
                assert_type(name, str)

    @assert_passes()
    def testTypeIsWithGenericInstance(self):
        from typing import Iterable, List, TypeVar

        from typing_extensions import TypeIs, assert_type

        _T = TypeVar("_T")

        def is_list_of_str(val: Iterable[_T]) -> TypeIs[List[_T]]:
            return False

        def func(name: Iterable[str]):
            assert_type(name, Iterable[str])
            if is_list_of_str(name):
                assert_type(name, List[str])

    @assert_passes()
    def testTypeIsWithTupleGeneric(self):
        from typing import Tuple, TypeVar

        from typing_extensions import TypeIs, assert_type

        _T = TypeVar("_T")

        def is_two_element_tuple(val: Tuple[_T, ...]) -> TypeIs[Tuple[_T, _T]]:
            return False

        def func(names: Tuple[str, ...]):
            assert_type(names, Tuple[str, ...])
            if is_two_element_tuple(names):
                assert_type(names, Tuple[str, ...])  # TODO: bad type narrowing

    @assert_passes()
    def testTypeIsErroneousDefinitionFails(self):
        from typing_extensions import TypeIs

        class Z:
            def typeguard1(self, *, x: object) -> TypeIs[int]:  # E: invalid_typeguard
                return False

            @staticmethod
            def typeguard2(x: object) -> TypeIs[int]:
                return False

            @staticmethod
            def typeguard3(*, x: object) -> TypeIs[int]:  # E: invalid_typeguard
                return False

        def bad_typeguard(*, x: object) -> TypeIs[int]:  # E: invalid_typeguard
            return False

    @assert_passes()
    def testTypeIsWithKeywordArg(self):
        from typing_extensions import TypeIs, assert_type

        class Z:
            def typeguard(self, x: object) -> TypeIs[int]:
                return False

        def typeguard(x: object) -> TypeIs[int]:
            return False

        def capybara(n: object) -> None:
            if typeguard(x=n):
                assert_type(n, int)

            if Z().typeguard(x=n):
                assert_type(n, int)

    @assert_passes()
    def testStaticMethodTypeIs(self):
        from typing_extensions import TypeIs, assert_type

        def typeguard(h: object) -> TypeIs[int]:
            return False

        class Y:
            @staticmethod
            def typeguard(h: object) -> TypeIs[int]:
                return False

        def capybara(x: object):
            if Y().typeguard(x):
                # This doesn't work because we treat it as a method, not a staticmethod,
                # and narrow parameter 1 instead. Doesn't look easy to fix, because the Signature
                # class has no way to know.
                assert_type(x, object)  # TODO: int
            assert_type(x, object)
            if Y.typeguard(x):
                assert_type(x, object)  # TODO: int

    @assert_passes()
    def testTypeIsKwargFollowingThroughOverloaded(self):
        from typing import Union, overload

        from typing_extensions import TypeIs, assert_type

        @overload
        def typeguard(x: object, y: str) -> TypeIs[str]: ...

        @overload
        def typeguard(x: object, y: int) -> TypeIs[int]: ...

        def typeguard(x: object, y: Union[int, str]) -> bool:
            return False

        def capybara(x: object) -> None:
            if typeguard(x=x, y=42):
                assert_type(x, int)

            if typeguard(y=42, x=x):
                assert_type(x, int)

            if typeguard(x=x, y="42"):
                assert_type(x, str)

            if typeguard(y="42", x=x):
                assert_type(x, str)

    @assert_passes()
    def testGenericAliasWithTypeIs(self):
        from typing import Callable, List, TypeVar

        from typing_extensions import TypeIs, assert_type

        T = TypeVar("T")
        A = Callable[[object], TypeIs[List[T]]]

        def foo(x: object) -> TypeIs[List[str]]:
            return False

        def test(f: A[T]) -> T:
            raise NotImplementedError

        def capybara() -> None:
            assert_type(test(foo), str)

    @assert_passes()
    def testNoCrashOnDunderCallTypeIs(self):
        from typing_extensions import TypeIs, assert_type

        class A:
            def __call__(self, x) -> TypeIs[int]:
                return True

        def capybara(a: A, x: object) -> None:
            assert a(x=1)

            assert a(x=x)
            # Seems like we drop the annotations on the __call__ return somewhere
            assert_type(x, object)  # TODO: int

    @assert_passes()
    def testTypeIsMustBeSubtypeFunctions(self):
        from typing import List, Sequence, TypeVar

        from typing_extensions import TypeIs

        def f(x: str) -> TypeIs[int]:  # E: typeis_must_be_subtype
            return False

        T = TypeVar("T")

        def g(x: List[T]) -> TypeIs[Sequence[T]]:  # E: typeis_must_be_subtype
            return False

    @assert_passes()
    def testTypeIsMustBeSubtypeMethods(self):
        from typing_extensions import TypeIs

        class NarrowHolder:
            @classmethod
            def cls_narrower_good(cls, x: object) -> TypeIs[int]:
                return False

            @classmethod
            def cls_narrower_bad(
                cls, x: str
            ) -> TypeIs[int]:  # E: typeis_must_be_subtype
                return False

            @staticmethod
            def static_narrower_good(x: object) -> TypeIs[int]:
                return False

            @staticmethod
            def static_narrower_bad(x: str) -> TypeIs[int]:  # E: typeis_must_be_subtype
                return False

            def inst_narrower_good(self, x: object) -> TypeIs[int]:
                return False

            def inst_narrower_bad(
                self, x: str
            ) -> TypeIs[int]:  # E: typeis_must_be_subtype
                return False
