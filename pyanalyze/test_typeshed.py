# static analysis: ignore
from qcore.asserts import assert_eq, assert_is, assert_is_instance
from qcore.testing import Anything
import collections.abc
from collections.abc import MutableSequence, Sequence, Collection, Reversible, Set
import contextlib
import io
import itertools
from pathlib import Path
import tempfile
import time
from typeshed_client import Resolver, get_search_context
import typing
from typing import Generic, TypeVar, NewType

from .test_config import TestConfig
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .signature import SigParameter, Signature
from .arg_spec import ArgSpecCache
from .test_arg_spec import ClassWithCall
from .typeshed import TypeshedFinder
from .value import (
    KnownValue,
    NewTypeValue,
    TypedValue,
    GenericValue,
    TypeVarValue,
    UNINITIALIZED_VALUE,
    UNRESOLVED_VALUE,
    SequenceIncompleteValue,
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
        # make sure this doesn't crash (it's defined as a function in typeshed)
        assert_is(None, tsf.get_bases(itertools.zip_longest))

    def test_newtype(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            (temp_dir / "typing.pyi").write_text("def NewType(a, b): pass\n")
            (temp_dir / "newt.pyi").write_text(
                """
from typing import NewType

NT = NewType("NT", int)

def f(x: NT) -> None:
    pass
"""
            )
            (temp_dir / "VERSIONS").write_text("newt: 3.5\ntyping: 3.5\n")
            tsf = TypeshedFinder(verbose=True)
            search_context = get_search_context(typeshed=temp_dir, search_path=[])
            tsf.resolver = Resolver(search_context)

            def runtime_f():
                pass

            sig = tsf.get_argspec_for_fully_qualified_name("newt.f", runtime_f)
            ntv = next(iter(tsf._assignment_cache.values()))
            assert_is_instance(ntv, NewTypeValue)
            assert_eq("NT", ntv.name)
            assert_eq(int, ntv.typ)
            assert_eq(
                Signature.make(
                    [SigParameter("x", annotation=ntv)],
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
                GenericValue(dict, [TypedValue(int), TypedValue(str)]),
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
        assert_eq({Parent: [UNRESOLVED_VALUE]}, self.get_generic_bases(Parent))
        assert_eq(
            {Parent: [TypeVarValue(T)]},
            self.get_generic_bases(Parent, [TypeVarValue(T)]),
        )
        assert_eq({Child: [], Parent: [TypedValue(int)]}, self.get_generic_bases(Child))
        assert_eq(
            {GenericChild: [UNRESOLVED_VALUE], Parent: [UNRESOLVED_VALUE]},
            self.get_generic_bases(GenericChild),
        )
        one = KnownValue(1)
        assert_eq(
            {GenericChild: [one], Parent: [one]},
            self.get_generic_bases(GenericChild, [one]),
        )

    def test_callable(self):
        assert_eq(
            {collections.abc.Callable: [], object: []},
            self.get_generic_bases(collections.abc.Callable, []),
        )

    def test_struct_time(self):
        assert_eq(
            {
                time.struct_time: [],
                # Ideally should be not Any, but we haven't implemented
                # support for typeshed namedtuples.
                tuple: [UNRESOLVED_VALUE],
                collections.abc.Collection: [UNRESOLVED_VALUE],
                collections.abc.Reversible: [UNRESOLVED_VALUE],
                collections.abc.Iterable: [UNRESOLVED_VALUE],
                collections.abc.Sequence: [UNRESOLVED_VALUE],
                collections.abc.Container: [UNRESOLVED_VALUE],
            },
            self.get_generic_bases(time.struct_time, []),
        )

    def test_context_manager(self):
        int_tv = TypedValue(int)
        assert_eq(
            {contextlib.AbstractContextManager: [int_tv]},
            self.get_generic_bases(contextlib.AbstractContextManager, [int_tv]),
        )

    def test_collections(self):
        int_tv = TypedValue(int)
        str_tv = TypedValue(str)
        int_str_tuple = SequenceIncompleteValue(tuple, [int_tv, str_tv])
        assert_eq(
            {
                collections.abc.ValuesView: [int_tv],
                collections.abc.MappingView: [],
                collections.abc.Iterable: [int_tv],
                collections.abc.Sized: [],
            },
            self.get_generic_bases(collections.abc.ValuesView, [int_tv]),
        )
        assert_eq(
            {
                collections.abc.ItemsView: [int_tv, str_tv],
                collections.abc.MappingView: [],
                collections.abc.Sized: [],
                collections.abc.Set: [int_str_tuple],
                collections.abc.Collection: [int_str_tuple],
                collections.abc.Iterable: [int_str_tuple],
                collections.abc.Container: [int_str_tuple],
            },
            self.get_generic_bases(collections.abc.ItemsView, [int_tv, str_tv]),
        )

        assert_eq(
            {
                collections.deque: [int_tv],
                collections.abc.MutableSequence: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Reversible: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Sequence: [int_tv],
                collections.abc.Container: [int_tv],
            },
            self.get_generic_bases(collections.deque, [int_tv]),
        )
        assert_eq(
            {
                collections.defaultdict: [int_tv, str_tv],
                dict: [int_tv, str_tv],
                collections.abc.MutableMapping: [int_tv, str_tv],
                collections.abc.Mapping: [int_tv, str_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [int_tv],
            },
            self.get_generic_bases(collections.defaultdict, [int_tv, str_tv]),
        )

    def test_typeshed(self):
        int_tv = TypedValue(int)
        str_tv = TypedValue(str)
        assert_eq(
            {
                list: [int_tv],
                collections.abc.MutableSequence: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Reversible: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Sequence: [int_tv],
                collections.abc.Container: [int_tv],
            },
            self.get_generic_bases(list, [int_tv]),
        )
        assert_eq(
            {
                set: [int_tv],
                collections.abc.MutableSet: [int_tv],
                collections.abc.Set: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [int_tv],
            },
            self.get_generic_bases(set, [int_tv]),
        )
        assert_eq(
            {
                dict: [int_tv, str_tv],
                collections.abc.MutableMapping: [int_tv, str_tv],
                collections.abc.Mapping: [int_tv, str_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [int_tv],
            },
            self.get_generic_bases(dict, [int_tv, str_tv]),
        )

    def test_io(self):
        assert_eq(
            {
                io.BytesIO: [],
                io.BufferedIOBase: [],
                io.IOBase: [],
                typing.BinaryIO: [],
                typing.IO: [TypedValue(bytes)],
                collections.abc.Iterator: [TypedValue(bytes)],
                collections.abc.Iterable: [TypedValue(bytes)],
            },
            self.get_generic_bases(io.BytesIO, []),
        )
