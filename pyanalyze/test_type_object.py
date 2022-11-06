# static analysis: ignore

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import (
    AnySource,
    AnyValue,
    assert_is_value,
    CallableValue,
    GenericValue,
    KnownValue,
    TypedValue,
)


class TestNumerics(TestNameCheckVisitorBase):
    @assert_passes()
    def test_float(self):
        from typing import NewType

        NT = NewType("NT", int)

        def take_float(x: float) -> None:
            pass

        def capybara(nt: NT, i: int, f: float) -> None:
            take_float(nt)
            take_float(i)
            take_float(f)
            take_float(3.0)
            take_float(3)
            take_float(1 + 1j)  # E: incompatible_argument
            take_float("string")  # E: incompatible_argument
            # Strictly speaking this should be allowed, but it doesn't
            # seem useful and I don't know of a concrete use case that would
            # require this to work. This can be revisited if we do find a use
            # case.
            take_float(True)  # E: incompatible_argument

    @assert_passes()
    def test_complex(self):
        from typing import NewType

        NTI = NewType("NTI", int)
        NTF = NewType("NTF", float)

        def take_complex(c: complex) -> None:
            pass

        def capybara(nti: NTI, ntf: NTF, i: int, f: float, c: complex) -> None:
            take_complex(ntf)
            take_complex(nti)
            take_complex(i)
            take_complex(f)
            take_complex(c)
            take_complex(3.0)
            take_complex(3)
            take_complex(1 + 1j)
            take_complex("string")  # E: incompatible_argument
            take_complex(True)  # E: incompatible_argument


class TestSyntheticType(TestNameCheckVisitorBase):
    @assert_passes()
    def test_functools(self):
        import functools
        import types

        from pyanalyze.signature import ELLIPSIS_PARAM, Signature

        sig = Signature.make([ELLIPSIS_PARAM], return_annotation=TypedValue(int))

        def f() -> int:
            return 0

        def capybara():
            c = functools.singledispatch(f)
            assert_is_value(
                c, GenericValue("functools._SingleDispatchCallable", [TypedValue(int)])
            )
            assert_is_value(
                c.registry,
                GenericValue(
                    types.MappingProxyType,
                    [AnyValue(AnySource.explicit), CallableValue(sig)],
                ),
            )
            assert_is_value(c._clear_cache(), KnownValue(None))
            assert_is_value(c(), TypedValue(int))
            c.doesnt_exist  # E: undefined_attribute

    @assert_passes()
    def test_protocol(self):
        # Note that csv.writer expects this protocol:
        # class _Writer(Protocol):
        #    def write(self, s: str) -> Any: ...
        import csv
        import io

        class BadWrite:
            def write(self, s: int) -> object:
                return object()

        class GoodWrite:
            def write(self, s: str) -> object:
                return object()

        class BadArgKind:
            def write(self, *, s: str) -> object:
                return object()

        def capybara(s: str):
            writer = io.StringIO()
            assert_is_value(csv.writer(writer), TypedValue("_csv._writer"))

            csv.writer(1)  # E: incompatible_argument
            csv.writer(s)  # E: incompatible_argument
            csv.writer(BadWrite())  # E: incompatible_argument
            csv.writer(GoodWrite())
            csv.writer(BadArgKind())  # E: incompatible_argument

    @assert_passes()
    def test_custom_subclasscheck(self):
        class _ThriftEnumMeta(type):
            def __subclasscheck__(self, subclass):
                return hasattr(subclass, "_VALUES_TO_NAMES")

        class ThriftEnum(metaclass=_ThriftEnumMeta):
            pass

        class IsOne:
            _VALUES_TO_NAMES = {}

        class IsntOne:
            _NAMES_TO_VALUES = {}

        def want_enum(te: ThriftEnum) -> None:
            pass

        def capybara(good_instance: IsOne, bad_instance: IsntOne, te: ThriftEnum):
            want_enum(good_instance)
            want_enum(bad_instance)  # E: incompatible_argument
            want_enum(IsOne())
            want_enum(IsntOne())  # E: incompatible_argument
            want_enum(te)

    @assert_passes()
    def test_generic_stubonly(self):
        import pkgutil

        # pkgutil.read_code requires SupportsRead[bytes]

        class Good:
            def read(self, length: int = 0) -> bytes:
                return b""

        class Bad:
            def read(self, length: int = 0) -> str:
                return ""

        def capybara():
            pkgutil.read_code(1)  # E: incompatible_argument
            pkgutil.read_code(Good())
            pkgutil.read_code(Bad())  # E: incompatible_argument

    @assert_passes()
    def test_protocol_inheritance(self):
        import cgi

        # cgi.parse requires SupportsItemAccess[str, str]

        class Good:
            def __contains__(self, obj: object) -> bool:
                return False

            def __getitem__(self, k: str) -> str:
                raise KeyError(k)

            def __setitem__(self, k: str, v: str) -> None:
                pass

            def __delitem__(self, v: str) -> None:
                pass

        class Bad:
            def __contains__(self, obj: object) -> bool:
                return False

            def __getitem__(self, k: bytes) -> str:
                raise KeyError(k)

            def __setitem__(self, k: str, v: str) -> None:
                pass

            def __delitem__(self, v: str) -> None:
                pass

        def capybara():
            cgi.parse(environ=Good())
            cgi.parse(environ=Bad())  # E: incompatible_argument
            cgi.parse(environ=1)  # E: incompatible_argument

    @assert_passes()
    def test_iterable(self):
        from typing import Iterable, Iterator

        class Bad:
            def __iter__(self, some, random, args):
                pass

        class Good:
            def __iter__(self) -> Iterator[int]:
                raise NotImplementedError

        class BadType:
            def __iter__(self) -> Iterator[str]:
                raise NotImplementedError

        def want_iter_int(f: Iterable[int]) -> None:
            pass

        def capybara():
            want_iter_int(Bad())  # E: incompatible_argument
            want_iter_int(Good())
            want_iter_int(BadType())  # E: incompatible_argument

    @assert_passes()
    def test_self_iterator(self):
        from typing import Iterator

        class MyIter:
            def __iter__(self) -> "MyIter":
                return self

            def __next__(self) -> int:
                return 42

        def want_iter(it: Iterator[int]):
            pass

        def capybara():
            want_iter(MyIter())

    @assert_passes()
    def test_container(self):
        from typing import Any, Container

        class Good:
            def __contains__(self, whatever: object) -> bool:
                return False

        class Bad:
            def __contains__(self, too, many, arguments) -> bool:
                return True

        def want_container(c: Container[Any]) -> None:
            pass

        def capybara() -> None:
            want_container(Bad())  # E: incompatible_argument
            want_container(Good())
            want_container([1])
            want_container(1)  # E: incompatible_argument

    @assert_passes()
    def test_runtime_protocol(self):
        from typing_extensions import Protocol

        class P(Protocol):
            a: int

            def b(self) -> int:
                raise NotImplementedError

        class Q(P, Protocol):
            c: str

        class NotAProtocol(P):
            c: str

        def want_p(x: P):
            print(x.a + x.b())

        def want_q(q: Q):
            pass

        def want_not_a_proto(nap: NotAProtocol):
            pass

        class GoodP:
            a: int

            def b(self) -> int:
                return 3

        class BadP:
            def a(self) -> int:
                return 5

            def b(self) -> int:
                return 4

        class GoodQ(GoodP):
            c: str

        class BadQ(GoodP):
            c: float

        def capybara():
            want_p(GoodP())
            want_p(BadP())  # E: incompatible_argument
            want_q(GoodQ())
            want_q(BadQ())  # E: incompatible_argument
            want_not_a_proto(GoodQ())  # E: incompatible_argument

    @assert_passes()
    def test_callable_protocol(self):
        from typing_extensions import Protocol

        class P(Protocol):
            def __call__(self, x: int) -> str:
                return str(x)

        def want_p(p: P) -> str:
            return p(1)

        def good(x: int) -> str:
            return "hello"

        def bad(x: str) -> str:
            return x

        def capybara():
            want_p(good)
            want_p(bad)  # E: incompatible_argument


class TestHashable(TestNameCheckVisitorBase):
    @assert_passes()
    def test_type(self):
        from typing import Hashable, Type

        from typing_extensions import Protocol

        class MyHashable(Protocol):
            def __hash__(self) -> int:
                raise NotImplementedError

        def want_hash(h: Hashable):
            pass

        def want_myhash(h: MyHashable):
            pass

        class A:
            pass

        def capybara(t1: Type[int], t2: type):
            want_hash(t1)
            want_hash(t2)
            want_hash(int)
            want_hash(A)

            # TODO these errors are wrong
            want_myhash(t1)
            want_myhash(t2)
            want_myhash(int)
            want_myhash(A)  # E: incompatible_argument

            {t1: 0}
            {t2: 0}
            {int: 0}
            {A: 0}

            want_hash([])  # E: incompatible_argument
            want_myhash([])  # E: incompatible_argument
