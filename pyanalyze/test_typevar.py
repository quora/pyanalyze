# static analysis: ignore
from .implementation import assert_is_value
from .value import (
    AnySource,
    AnyValue,
    KnownValue,
    MultiValuedValue,
    TypedValue,
    AnnotatedValue,
)
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before


class TestTypeVar(TestNameCheckVisitorBase):
    @assert_passes()
    def test_simple(self):
        from typing import TypeVar, List, Generic

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
        from typing import TypeVar, Optional

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
        from typing import TypeVar, Optional

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
    def test_generic_base(self):
        from typing import TypeVar, Generic

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
        from typing import TypeVar, Generic

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
        from typing import TypeVar, Dict, Union

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
        from typing import TypeVar, Sequence, Union

        AnyStr = TypeVar("AnyStr", bound=Union[str, bytes])

        def take_seq(seq: Sequence[AnyStr]) -> Sequence[AnyStr]:
            return seq

        def take_union(seq: Union[Sequence[bytes], Sequence[str]]) -> None:
            take_seq(seq)

    @assert_passes()
    def test_call_with_value_restriction(self):
        from typing import TypeVar, Callable, Union

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


class TestAnnotated(TestNameCheckVisitorBase):
    @assert_passes()
    def test_preserve(self):
        from typing_extensions import Annotated
        from typing import TypeVar

        T = TypeVar("T")

        def f(x: T) -> T:
            return x

        def caller(x: Annotated[int, 42]):
            assert_is_value(x, AnnotatedValue(TypedValue(int), [KnownValue(42)]))
            assert_is_value(f(x), AnnotatedValue(TypedValue(int), [KnownValue(42)]))
