# static analysis: ignore
from qcore.asserts import assert_eq, assert_is, assert_is_instance
from qcore.testing import Anything
import collections.abc
from collections.abc import MutableSequence, Sequence, Collection, Reversible, Set
import contextlib
import io
from pathlib import Path
import tempfile
import time
from typeshed_client import Resolver, get_search_context
import typing
from typing import Dict, Generic, List, TypeVar, NewType
from urllib.error import HTTPError

from .test_config import TestConfig
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .signature import SigParameter, Signature
from .arg_spec import ArgSpecCache
from .test_arg_spec import ClassWithCall
from .typeshed import TypeshedFinder
from .value import (
    assert_is_value,
    AnySource,
    AnyValue,
    KnownValue,
    NewTypeValue,
    TypedValue,
    GenericValue,
    make_weak,
    TypeVarValue,
    UNINITIALIZED_VALUE,
    SequenceIncompleteValue,
    Value,
)

T = TypeVar("T")
NT = NewType("NT", int)


class TestTypeshedClient(TestNameCheckVisitorBase):
    @assert_passes()
    def test_types(self):
        import math

        assert_is_value(math.exp(1.0), TypedValue(float))
        assert_is_value("".isspace(), TypedValue(bool))

    @assert_passes()
    def test_dict_update(self):
        def capybara():
            x = {}
            x.update({})  # just check that this doesn't fail

    def test_get_bases(self):
        tsf = TypeshedFinder(verbose=True)
        assert_eq(
            [
                GenericValue(MutableSequence, (TypeVarValue(typevar=Anything),)),
                GenericValue(Generic, (TypeVarValue(typevar=Anything),)),
            ],
            tsf.get_bases(list),
        )
        assert_eq(
            [
                GenericValue(Collection, (TypeVarValue(typevar=Anything),)),
                GenericValue(Reversible, (TypeVarValue(typevar=Anything),)),
                GenericValue(Generic, (TypeVarValue(typevar=Anything),)),
            ],
            tsf.get_bases(Sequence),
        )
        assert_eq(
            [GenericValue(Collection, (TypeVarValue(Anything),))], tsf.get_bases(Set)
        )

    def test_newtype(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            (temp_dir / "typing.pyi").write_text("def NewType(a, b): pass\n")
            (temp_dir / "newt.pyi").write_text(
                """
from typing import NewType

NT = NewType("NT", int)
Alias = int

def f(x: NT, y: Alias) -> None:
    pass
"""
            )
            (temp_dir / "VERSIONS").write_text("newt: 3.5\ntyping: 3.5\n")
            (temp_dir / "@python2").mkdir()
            tsf = TypeshedFinder(verbose=True)
            search_context = get_search_context(typeshed=temp_dir, search_path=[])
            tsf.resolver = Resolver(search_context)

            def runtime_f():
                pass

            sig = tsf.get_argspec_for_fully_qualified_name("newt.f", runtime_f)
            newtype = next(iter(tsf._assignment_cache.values()))
            assert_is_instance(newtype, KnownValue)
            ntv = NewTypeValue(newtype.val)
            assert_eq("NT", ntv.name)
            assert_eq(int, ntv.typ)
            assert_eq(
                Signature.make(
                    [
                        SigParameter("x", annotation=ntv),
                        SigParameter("y", annotation=TypedValue(int)),
                    ],
                    KnownValue(None),
                    callable=runtime_f,
                ),
                sig,
            )

    @assert_passes()
    def test_generic_self(self):
        from typing import Dict

        def capybara(x: Dict[int, str]):
            assert_is_value(
                {k: v for k, v in x.items()},
                make_weak(GenericValue(dict, [TypedValue(int), TypedValue(str)])),
            )

    @assert_passes()
    def test_str_find(self):
        def capybara(s: str) -> None:
            assert_is_value(s.find("x"), TypedValue(int))

    def test_has_stubs(self) -> None:
        tsf = TypeshedFinder(verbose=True)
        assert tsf.has_stubs(object)
        assert not tsf.has_stubs(ClassWithCall)

    def test_get_attribute(self) -> None:
        tsf = TypeshedFinder(verbose=True)
        assert_is(UNINITIALIZED_VALUE, tsf.get_attribute(object, "nope"))
        assert_eq(
            TypedValue(bool), tsf.get_attribute(staticmethod, "__isabstractmethod__")
        )


class Parent(Generic[T]):
    pass


class Child(Parent[int]):
    pass


class GenericChild(Parent[T]):
    pass


class TestGetGenericBases:
    def setup(self) -> None:
        arg_spec_cache = ArgSpecCache(TestConfig())
        self.get_generic_bases = arg_spec_cache.get_generic_bases

    def test_runtime(self):
        assert_eq(
            {Parent: {T: AnyValue(AnySource.generic_argument)}},
            self.get_generic_bases(Parent),
        )
        assert_eq(
            {Parent: {T: TypeVarValue(T)}},
            self.get_generic_bases(Parent, [TypeVarValue(T)]),
        )
        assert_eq(
            {Child: {}, Parent: {T: TypedValue(int)}}, self.get_generic_bases(Child)
        )
        assert_eq(
            {
                GenericChild: {T: AnyValue(AnySource.generic_argument)},
                Parent: {T: AnyValue(AnySource.generic_argument)},
            },
            self.get_generic_bases(GenericChild),
        )
        one = KnownValue(1)
        assert_eq(
            {GenericChild: {T: one}, Parent: {T: one}},
            self.get_generic_bases(GenericChild, [one]),
        )

    def check(
        self,
        expected: Dict[type, List[Value]],
        base: type,
        args: typing.Sequence[Value] = (),
    ) -> None:
        actual = self.get_generic_bases(base, args)
        cleaned = {base: list(tv_map.values()) for base, tv_map in actual.items()}
        assert_eq(expected, cleaned, extra=actual)

    def test_coroutine(self):
        one = KnownValue(1)
        two = KnownValue(2)
        three = KnownValue(3)
        self.check(
            {
                collections.abc.Coroutine: [one, two, three],
                collections.abc.Awaitable: [three],
            },
            collections.abc.Coroutine,
            [one, two, three],
        )

    def test_callable(self):
        self.check({collections.abc.Callable: []}, collections.abc.Callable)

    def test_struct_time(self):
        self.check(
            {
                time.struct_time: [],
                # Ideally should be not Any, but we haven't implemented
                # support for typeshed namedtuples.
                tuple: [AnyValue(AnySource.generic_argument)],
                collections.abc.Collection: [AnyValue(AnySource.generic_argument)],
                collections.abc.Reversible: [AnyValue(AnySource.generic_argument)],
                collections.abc.Iterable: [AnyValue(AnySource.generic_argument)],
                collections.abc.Sequence: [AnyValue(AnySource.generic_argument)],
                collections.abc.Container: [AnyValue(AnySource.generic_argument)],
            },
            time.struct_time,
        )

    def test_context_manager(self):
        int_tv = TypedValue(int)
        self.check(
            {contextlib.AbstractContextManager: [int_tv]},
            contextlib.AbstractContextManager,
            [int_tv],
        )

    def test_collections(self):
        int_tv = TypedValue(int)
        str_tv = TypedValue(str)
        int_str_tuple = SequenceIncompleteValue(tuple, [int_tv, str_tv])
        self.check(
            {
                collections.abc.ValuesView: [int_tv],
                collections.abc.MappingView: [],
                collections.abc.Iterable: [int_tv],
                collections.abc.Sized: [],
            },
            collections.abc.ValuesView,
            [int_tv],
        )
        self.check(
            {
                collections.abc.ItemsView: [int_tv, str_tv],
                collections.abc.MappingView: [],
                collections.abc.Sized: [],
                collections.abc.Set: [int_str_tuple],
                collections.abc.Collection: [int_str_tuple],
                collections.abc.Iterable: [int_str_tuple],
                collections.abc.Container: [int_str_tuple],
            },
            collections.abc.ItemsView,
            [int_tv, str_tv],
        )

        self.check(
            {
                collections.deque: [int_tv],
                collections.abc.MutableSequence: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Reversible: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Sequence: [int_tv],
                collections.abc.Container: [int_tv],
            },
            collections.deque,
            [int_tv],
        )
        self.check(
            {
                collections.defaultdict: [int_tv, str_tv],
                dict: [int_tv, str_tv],
                collections.abc.MutableMapping: [int_tv, str_tv],
                collections.abc.Mapping: [int_tv, str_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [int_tv],
            },
            collections.defaultdict,
            [int_tv, str_tv],
        )

    def test_typeshed(self):
        int_tv = TypedValue(int)
        str_tv = TypedValue(str)
        self.check(
            {
                list: [int_tv],
                collections.abc.MutableSequence: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Reversible: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Sequence: [int_tv],
                collections.abc.Container: [int_tv],
            },
            list,
            [int_tv],
        )
        self.check(
            {
                set: [int_tv],
                collections.abc.MutableSet: [int_tv],
                collections.abc.Set: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [int_tv],
            },
            set,
            [int_tv],
        )
        self.check(
            {
                dict: [int_tv, str_tv],
                collections.abc.MutableMapping: [int_tv, str_tv],
                collections.abc.Mapping: [int_tv, str_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [int_tv],
            },
            dict,
            [int_tv, str_tv],
        )

    def test_io(self):
        self.check(
            {
                io.BytesIO: [],
                io.BufferedIOBase: [],
                io.IOBase: [],
                typing.BinaryIO: [],
                typing.IO: [TypedValue(bytes)],
                collections.abc.Iterator: [TypedValue(bytes)],
                collections.abc.Iterable: [TypedValue(bytes)],
            },
            io.BytesIO,
        )


class TestAttribute:
    def test_basic(self) -> None:
        tsf = TypeshedFinder(verbose=True)
        assert_eq(
            TypedValue(bool), tsf.get_attribute(staticmethod, "__isabstractmethod__")
        )

    def test_property(self) -> None:
        tsf = TypeshedFinder(verbose=True)
        assert_eq(TypedValue(int), tsf.get_attribute(int, "real"))

    def test_http_error(self) -> None:
        tsf = TypeshedFinder(verbose=True)
        assert_is(True, tsf.has_attribute(HTTPError, "read"))
