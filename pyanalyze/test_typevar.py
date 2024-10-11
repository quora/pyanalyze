# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before
from .tests import make_simple_sequence
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    GenericValue,
    KnownValue,
    MultiValuedValue,
    TypedValue,
)


class TestTypeVar(TestNameCheckVisitorBase):
    @assert_passes()
    def test_simple(self):
        from typing import Generic, List, TypeVar

        T = TypeVar("T")

        def id(obj: T) -> T:
            return obj

        def get_one(obj: List[T]) -> T:
            for elt in obj:
                return elt
            assert False

        class GenCls(Generic[T]):
            def get_one(self: "GenCls[T]") -> T:
                raise NotImplementedError

            def get_another(self) -> T:
                raise NotImplementedError

        def capybara(x: str, xs: List[int], gen: GenCls[int]) -> None:
            assert_is_value(id(3), KnownValue(3))
            assert_is_value(id(x), TypedValue(str))
            assert_is_value(get_one(xs), TypedValue(int))
            assert_is_value(get_one([int(3)]), TypedValue(int))
            # This one doesn't work yet because we don't know how to go from
            # KnownValue([3]) to a GenericValue of some sort.
            # assert_is_value(get_one([3]), KnownValue(3))

            assert_is_value(gen.get_one(), TypedValue(int))
            assert_is_value(gen.get_another(), TypedValue(int))

    @assert_passes()
    def test_union_math(self):
        from typing import Optional, TypeVar

        T = TypeVar("T")

        def assert_not_none(arg: Optional[T]) -> T:
            assert arg is not None
            return arg

        def capybara(x: Optional[int]):
            assert_is_value(x, MultiValuedValue([KnownValue(None), TypedValue(int)]))
            assert_is_value(assert_not_none(x), TypedValue(int))

    @assert_passes()
    def test_only_T(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Capybara(Generic[T]):
            def add_one(self, obj: T) -> None:
                pass

        def capybara(x: Capybara[int]) -> None:
            x.add_one("x")  # E: incompatible_argument

    @assert_passes()
    def test_multi_typevar(self):
        from typing import Optional, TypeVar

        T = TypeVar("T")

        # inspired by tempfile.mktemp
        def mktemp(prefix: Optional[T] = None, suffix: Optional[T] = None) -> T:
            raise NotImplementedError

        def capybara() -> None:
            assert_is_value(mktemp(), AnyValue(AnySource.generic_argument))
            assert_is_value(mktemp(prefix="p"), KnownValue("p"))
            assert_is_value(mktemp(suffix="s"), KnownValue("s"))
            assert_is_value(mktemp("p", "s"), KnownValue("p") | KnownValue("s"))

    @assert_passes()
    def test_generic_constructor(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Capybara(Generic[T]):
            x: T

            def __init__(self, x: T) -> None:
                self.x = x

        def capybara(i: int) -> None:
            assert_is_value(Capybara(i).x, TypedValue(int))

    @assert_passes()
    def test_generic_base(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Base(Generic[T]):
            pass

        class Derived(Base[int]):
            pass

        def take_base(b: Base[int]) -> None:
            pass

        def capybara(c: Derived):
            take_base(c)

    @assert_passes()
    def test_wrong_generic_base(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Base(Generic[T]):
            pass

        class Derived(Base[int]):
            pass

        def take_base(b: Base[str]) -> None:
            pass

        def capybara(c: Derived):
            take_base(c)  # E: incompatible_argument

    @skip_before((3, 10))
    @assert_passes()
    def test_typeshed(self):
        from typing import List

        def capybara(lst: List[int]) -> None:
            lst.append("x")  # E: incompatible_argument

    @assert_passes()
    def test_generic_super(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class A(Generic[T]):
            def capybara(self) -> None:
                pass

        class B(A):
            def capybara(self) -> None:
                super().capybara()

    @assert_passes()
    def test_default(self):
        from typing import Dict, TypeVar, Union

        KT = TypeVar("KT")
        VT = TypeVar("VT")
        T = TypeVar("T")

        def dictget(d: Dict[KT, VT], key: KT, default: T = None) -> Union[VT, T]:
            try:
                return d[key]
            except KeyError:
                return default

        def capybara(d: Dict[str, str], key: str) -> None:
            assert_is_value(dictget(d, key), TypedValue(str) | KnownValue(None))
            assert_is_value(dictget(d, key, 1), TypedValue(str) | KnownValue(1))


class TestSolve(TestNameCheckVisitorBase):
    @assert_passes()
    def test_filter_like(self):
        from typing import Callable, TypeVar

        T = TypeVar("T")

        def callable(o: object) -> bool:
            return True

        def filterish(func: Callable[[T], bool], data: T) -> T:
            return data

        def capybara():
            assert_is_value(filterish(callable, 1), KnownValue(1))

    @assert_passes()
    def test_one_any(self):
        from typing import TypeVar

        T = TypeVar("T")

        def sub(x: T, y: T) -> T:
            return x

        def capybara(unannotated):
            assert_is_value(sub(1, unannotated), KnownValue(1))

    @assert_passes()
    def test_isinstance(self):
        from typing import TypeVar

        AnyStr = TypeVar("AnyStr", str, bytes)

        class StrSub(str):
            pass

        def want_str(s: StrSub) -> None:
            pass

        def take_tv(t: AnyStr) -> AnyStr:
            if isinstance(t, StrSub):
                assert_is_value(t, TypedValue(StrSub))
                return t
            else:
                return t

    @assert_passes()
    def test_tv_union(self):
        from typing import TypeVar, Union

        AnyStr = TypeVar("AnyStr", str, bytes)

        def take_seq(seq: AnyStr) -> AnyStr:
            return seq

        def take_union(seq: Union[bytes, str]) -> None:
            take_seq(seq)  # E: incompatible_argument

    @assert_passes()
    def test_tv_sequence(self):
        from typing import Sequence, TypeVar, Union

        AnyStr = TypeVar("AnyStr", bound=Union[str, bytes])

        def take_seq(seq: Sequence[AnyStr]) -> Sequence[AnyStr]:
            return seq

        def take_union(seq: Union[Sequence[bytes], Sequence[str]]) -> None:
            take_seq(seq)

    @assert_passes()
    def test_call_with_value_restriction(self):
        from typing import Callable, TypeVar, Union

        CallableT = TypeVar("CallableT", Callable[..., str], Callable[..., int])
        UnionT = TypeVar("UnionT", bound=Union[Callable[..., str], Callable[..., int]])

        def capybara(c: CallableT, u: UnionT) -> None:
            c(3)
            u(3)

    @assert_passes()
    def test_min_enum(self):
        import enum

        class E(enum.IntEnum):
            a = 1
            b = 2

        def capybara():
            m = min(E)
            assert_is_value(m, TypedValue(E))

    @assert_passes()
    def test_constraints(self):
        from typing import List, TypeVar

        LT = TypeVar("LT", List[int], List[str])

        def g(x: LT) -> LT:
            return x

        def pacarana() -> None:
            assert_is_value(g([]), AnyValue(AnySource.inference))
            assert_is_value(g([1]), GenericValue(list, [TypedValue(int)]))

    @assert_passes()
    def test_redundant_constraints(self):
        from typing import TypeVar

        from typing_extensions import SupportsIndex

        T = TypeVar("T", int, float, SupportsIndex)

        def f(x: T) -> T:
            return x

        def capybara(si: SupportsIndex):
            assert_is_value(f(1), TypedValue(int))
            assert_is_value(f(si), TypedValue(SupportsIndex))
            assert_is_value(f(1.0), TypedValue(float))

    @assert_passes()
    def test_lots_of_constraints(self):
        from typing import TypeVar, Union

        from typing_extensions import SupportsIndex

        T = TypeVar(
            "T",
            Union[int, str],
            Union[int, float],
            Union[int, range],
            Union[int, bytes],
            SupportsIndex,
            Union[int, bytearray],
            Union[int, memoryview],
            Union[int, list],
            Union[int, tuple],
            Union[int, set],
            Union[int, frozenset],
            Union[int, dict],
        )

        def f(x: T) -> T:
            return x

        def capybara(si: SupportsIndex):
            assert_is_value(f(1), AnyValue(AnySource.inference))

    @assert_passes()
    def test_or_bounds(self):
        from typing import Dict, Mapping, Tuple, TypeVar, Union

        T = TypeVar("T")
        U = TypeVar("U")

        def capybara(d: Union[Dict[T, U], Mapping[U, T]]) -> Tuple[T, U]:
            raise NotImplementedError

        def caller():
            result = capybara({"x": 1})
            assert_is_value(
                result,
                make_simple_sequence(
                    tuple,
                    [
                        AnyValue(AnySource.generic_argument),
                        AnyValue(AnySource.generic_argument),
                    ],
                ),
            )


class TestAnnotated(TestNameCheckVisitorBase):
    @assert_passes()
    def test_preserve(self):
        from typing import TypeVar

        from typing_extensions import Annotated

        T = TypeVar("T")

        def f(x: T) -> T:
            return x

        def caller(x: Annotated[int, 42]):
            assert_is_value(x, AnnotatedValue(TypedValue(int), [KnownValue(42)]))
            assert_is_value(f(x), AnnotatedValue(TypedValue(int), [KnownValue(42)]))


class TestDunder(TestNameCheckVisitorBase):
    @assert_passes()
    def test_sequence(self):
        from typing import Sequence

        from typing_extensions import assert_type

        def capybara(s: Sequence[int], t: str):
            assert_type(s[0], int)


class TestGenericClasses(TestNameCheckVisitorBase):
    @skip_before((3, 12))
    def test_generic(self):
        self.assert_passes(
            """
            from typing_extensions import assert_type

            class C[T]:
                x: T

                def __init__(self, x: T) -> None:
                    self.x = x

            def capybara(i: int):
                assert_type(C(i).x, int)
        """
        )

    @skip_before((3, 12))
    def test_generic_with_bound(self):
        self.assert_passes(
            """
            from typing_extensions import assert_type

            class C[T: int]:
                x: T

                def __init__(self, x: T) -> None:
                    self.x = x

            def capybara(i: int, s: str, b: bool):
                assert_type(C(i).x, int)
                assert_type(C(b).x, bool)
                C(s)  # E: incompatible_argument
        """
        )
