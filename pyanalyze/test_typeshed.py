# static analysis: ignore
import collections
import collections.abc
import contextlib
import io
import sys
import tempfile
import textwrap
import time
import typing
import urllib.parse
from collections.abc import Collection, MutableSequence, Reversible, Sequence, Set
from pathlib import Path
from typing import Dict, Generic, List, NewType, Type, TypeVar, Union
from unittest.mock import ANY
from urllib.error import HTTPError

from typeshed_client import Resolver, get_search_context

from .checker import Checker
from .extensions import evaluated
from .signature import OverloadedSignature, Signature, SigParameter
from .test_arg_spec import ClassWithCall
from .test_config import TEST_OPTIONS
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .tests import make_simple_sequence
from .typeshed import TypeshedFinder
from .value import (
    UNINITIALIZED_VALUE,
    AnySource,
    AnyValue,
    CallableValue,
    DictIncompleteValue,
    GenericValue,
    KnownValue,
    KVPair,
    NewTypeValue,
    SequenceValue,
    SubclassValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeVarValue,
    Value,
    assert_is_value,
)

T = TypeVar("T")
NT = NewType("NT", int)


class TestTypeshedClient(TestNameCheckVisitorBase):
    @assert_passes()
    def test_types(self):
        import math
        from typing import Container

        assert_is_value(math.exp(1.0), TypedValue(float))
        assert_is_value("".isspace(), TypedValue(bool))

        def capybara(x: Container[int]) -> None:
            assert_is_value(x.__contains__(1), TypedValue(bool))

    @assert_passes()
    def test_dict_update(self):
        def capybara():
            x = {}
            x.update({})  # just check that this doesn't fail

    def test_get_bases(self):
        tsf = TypeshedFinder(Checker(), verbose=True)
        generic = GenericValue(Generic, (TypeVarValue(typevar=ANY),))

        # typeshed removed Generic[] from the base list, account for both options
        def assert_with_maybe_generic(cls: Type[object], expected: List[Value]) -> None:
            actual = tsf.get_bases(cls)
            assert actual == expected or actual == [*expected, generic]

        assert_with_maybe_generic(
            list, [GenericValue(MutableSequence, (TypeVarValue(typevar=ANY),))]
        )
        assert_with_maybe_generic(
            Sequence,
            [
                GenericValue(Collection, (TypeVarValue(typevar=ANY),)),
                GenericValue(Reversible, (TypeVarValue(typevar=ANY),)),
            ],
        )
        assert_with_maybe_generic(Set, [GenericValue(Collection, (TypeVarValue(ANY),))])

    def test_newtype(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            (temp_dir / "typing.pyi").write_text("def NewType(a, b): pass\n")
            (temp_dir / "newt.pyi").write_text(
                textwrap.dedent(
                    """
                from typing import NewType

                NT = NewType("NT", int)
                Alias = int

                def f(x: NT, y: Alias) -> None:
                    pass
                """
                )
            )
            (temp_dir / "VERSIONS").write_text("newt: 3.5\ntyping: 3.5\n")
            (temp_dir / "@python2").mkdir()
            tsf = TypeshedFinder(Checker(), verbose=True)
            search_context = get_search_context(typeshed=temp_dir, search_path=[])
            tsf.resolver = Resolver(search_context)

            def runtime_f():
                pass

            sig = tsf.get_argspec_for_fully_qualified_name("newt.f", runtime_f)
            newtype = next(iter(tsf._assignment_cache.values()))
            assert isinstance(newtype, KnownValue)
            ntv = NewTypeValue(newtype.val)
            assert "NT" == ntv.name
            assert int == ntv.typ
            assert (
                Signature.make(
                    [
                        SigParameter("x", annotation=ntv),
                        SigParameter("y", annotation=TypedValue(int)),
                    ],
                    KnownValue(None),
                    callable=runtime_f,
                )
                == sig
            )

    @assert_passes()
    def test_generic_self(self):
        from typing import Dict

        def capybara(x: Dict[int, str]):
            assert_is_value(
                {k: v for k, v in x.items()},
                DictIncompleteValue(
                    dict, [KVPair(TypedValue(int), TypedValue(str), is_many=True)]
                ),
            )

    @assert_passes()
    def test_str_find(self):
        def capybara(s: str) -> None:
            assert_is_value(s.find("x"), TypedValue(int))

    @assert_passes()
    def test_str_count(self):
        def capybara(s: str) -> None:
            assert_is_value(s.count("x"), TypedValue(int))

    @assert_passes()
    def test_dict_fromkeys(self):
        def capybara(i: int) -> None:
            assert_is_value(
                dict.fromkeys([i]),
                GenericValue(
                    dict,
                    [TypedValue(int), AnyValue(AnySource.explicit) | KnownValue(None)],
                ),
            )

    @assert_passes()
    def test_datetime(self):
        from datetime import datetime

        from typing_extensions import assert_type

        def capybara(i: int):
            dt = datetime.fromtimestamp(i)
            assert_type(dt, datetime)

    def test_has_stubs(self) -> None:
        tsf = TypeshedFinder(Checker(), verbose=True)
        assert tsf.has_stubs(object)
        assert not tsf.has_stubs(ClassWithCall)

    def test_get_attribute(self) -> None:
        tsf = TypeshedFinder(Checker(), verbose=True)
        assert UNINITIALIZED_VALUE is tsf.get_attribute(object, "nope", on_class=False)
        assert TypedValue(bool) == tsf.get_attribute(
            staticmethod, "__isabstractmethod__", on_class=False
        )


_EXPECTED_TYPED_DICTS = {
    "TD1": TypedDictValue(
        {"a": TypedDictEntry(TypedValue(int)), "b": TypedDictEntry(TypedValue(str))}
    ),
    "TD2": TypedDictValue(
        {
            "a": TypedDictEntry(TypedValue(int), required=False),
            "b": TypedDictEntry(TypedValue(str), required=False),
        }
    ),
    "PEP655": TypedDictValue(
        {
            "a": TypedDictEntry(TypedValue(int), required=False),
            "b": TypedDictEntry(TypedValue(str)),
        }
    ),
    "Inherited": TypedDictValue(
        {
            "a": TypedDictEntry(TypedValue(int)),
            "b": TypedDictEntry(TypedValue(str)),
            "c": TypedDictEntry(TypedValue(float)),
        }
    ),
}


class TestBundledStubs(TestNameCheckVisitorBase):
    @assert_passes()
    def test_import_aliases(self):
        def capybara():
            from _pyanalyze_tests.aliases import (
                ExplicitAlias,
                aliased_constant,
                constant,
                explicitly_aliased_constant,
            )

            assert_is_value(ExplicitAlias, KnownValue(int))
            assert_is_value(constant, TypedValue(int))
            assert_is_value(aliased_constant, TypedValue(int))
            assert_is_value(explicitly_aliased_constant, TypedValue(int))

    def test_aliases(self):
        tsf = TypeshedFinder.make(Checker(), TEST_OPTIONS, verbose=True)
        mod = "_pyanalyze_tests.aliases"
        assert tsf.resolve_name(mod, "constant") == TypedValue(int)
        assert tsf.resolve_name(mod, "aliased_constant") == TypedValue(int)
        assert tsf.resolve_name(mod, "explicitly_aliased_constant") == TypedValue(int)

    def test_overloaded(self):
        tsf = TypeshedFinder.make(Checker(), TEST_OPTIONS, verbose=True)
        mod = "_pyanalyze_tests.overloaded"
        val = tsf.resolve_name(mod, "func")
        assert isinstance(val, CallableValue)
        assert isinstance(val.signature, OverloadedSignature)

    @assert_passes()
    def test_pos_only(self):
        def capybara():
            from _pyanalyze_tests.posonly import f, g, h, two_pos_only

            f(1, 2)
            f(1, 2, 3)
            f(1, y=2, z=3)
            f(x=1, y=2)  # E: incompatible_call
            f()  # E: incompatible_call

            g()
            g(1)
            g(1, 2)
            g(1, y=2)
            g(x=1)  # E: incompatible_call

            h()  # E: incompatible_call
            h(1)
            h(x=1)  # E: incompatible_call
            h(1, y=2)
            h(1, 2, 3)

            two_pos_only(1, "x")
            two_pos_only(1)
            two_pos_only(x=1)  # E: incompatible_call
            two_pos_only(1, y="x")  # E: incompatible_call

    @assert_passes()
    def test_typevar_with_default(self):
        def capybara(x: int):
            from _pyanalyze_tests.typevar import f
            from typing_extensions import assert_type

            assert_type(f(x), int)

    @assert_passes()
    def test_typing_extensions_paramspec(self):
        def some_func(x: int) -> str:
            return str(x)

        def capybara(x: int):
            from _pyanalyze_tests.paramspec import f, g
            from typing_extensions import assert_type

            assert_type(f(x), int)
            assert_type(g(some_func, x), str)
            g(some_func, "not an int")  # TODO should error

    def test_typeddict(self):
        tsf = TypeshedFinder.make(Checker(), TEST_OPTIONS, verbose=True)
        mod = "_pyanalyze_tests.typeddict"

        for name, expected in _EXPECTED_TYPED_DICTS.items():
            assert tsf.resolve_name(mod, name) == SubclassValue(expected, exactly=True)

    @assert_passes()
    def test_cdata(self):
        def capybara():
            from _pyanalyze_tests.cdata import f

            assert_is_value(f(), TypedValue("_ctypes._CData"))

    @assert_passes()
    def test_ast(self):
        import ast

        def capybara(x: ast.Yield):
            assert_is_value(x, TypedValue(ast.Yield))
            assert_is_value(x.value, TypedValue(ast.expr) | KnownValue(None))

    @assert_passes()
    def test_import_typeddicts(self):
        def capybara():
            from _pyanalyze_tests.typeddict import PEP655, TD1, TD2, Inherited

            from pyanalyze.test_typeshed import _EXPECTED_TYPED_DICTS

            def nested(td1: TD1, td2: TD2, pep655: PEP655, inherited: Inherited):
                assert_is_value(td1, _EXPECTED_TYPED_DICTS["TD1"])
                assert_is_value(td2, _EXPECTED_TYPED_DICTS["TD2"])
                assert_is_value(pep655, _EXPECTED_TYPED_DICTS["PEP655"])
                assert_is_value(inherited, _EXPECTED_TYPED_DICTS["Inherited"])

    def test_evaluated(self):
        tsf = TypeshedFinder.make(Checker(), TEST_OPTIONS, verbose=True)
        mod = "_pyanalyze_tests.evaluated"
        assert tsf.resolve_name(mod, "evaluated") == KnownValue(evaluated)

    @assert_passes()
    def test_evaluated_import(self):
        def capybara(unannotated):
            from typing import IO, BinaryIO, TextIO

            from _pyanalyze_tests.evaluated import open, open2

            assert_is_value(open("r"), TypedValue(TextIO))
            assert_is_value(open("rb"), TypedValue(BinaryIO))
            assert_is_value(
                open(unannotated), GenericValue(IO, [AnyValue(AnySource.explicit)])
            )
            assert_is_value(
                open("r" if unannotated else "rb"),
                TypedValue(TextIO) | TypedValue(BinaryIO),
            )
            assert_is_value(open2("r"), TypedValue(TextIO))
            assert_is_value(open2("rb"), TypedValue(BinaryIO))
            assert_is_value(
                open2(unannotated), GenericValue(IO, [AnyValue(AnySource.explicit)])
            )
            assert_is_value(
                open2("r" if unannotated else "rb"),
                TypedValue(TextIO) | TypedValue(BinaryIO),
            )

    @assert_passes()
    def test_recursive_base(self):
        from typing import Any, ContextManager

        def capybara():
            from _pyanalyze_tests.recursion import _ScandirIterator

            def want_cm(cm: ContextManager[Any]) -> None:
                pass

            def f(x: _ScandirIterator):
                want_cm(x)
                len(x)  # E: incompatible_argument

    @assert_passes()
    def test_args_kwargs(self):
        def capybara():
            from _pyanalyze_tests.args import f, g, h, i

            f(1)  # E: incompatible_call
            f(1, "x")
            g(x=1)  # E: incompatible_call
            g(x=1, y="x")
            h("x")  # E: incompatible_argument
            h()
            h(1)
            i(x=3)  # E: incompatible_argument
            i(x="x")
            i()

    @assert_passes()
    def test_stub_context_manager(self):
        from typing_extensions import Literal, assert_type

        def capybara():
            from _pyanalyze_tests.contextmanager import cm

            with cm() as f:
                assert_type(f, int)
                x = 3

            assert_type(x, Literal[3])

    @assert_passes()
    def test_stub_defaults(self):
        def capybara():
            from _pyanalyze_tests.defaults import many_defaults

            a, b, c, d = many_defaults()
            assert_is_value(
                a, DictIncompleteValue(dict, [KVPair(KnownValue("a"), KnownValue(1))])
            )
            assert_is_value(
                b,
                SequenceValue(
                    list, [(False, KnownValue(1)), (False, SequenceValue(tuple, []))]
                ),
            )
            assert_is_value(
                c,
                SequenceValue(tuple, [(False, KnownValue(1)), (False, KnownValue(2))]),
            )
            assert_is_value(
                d, SequenceValue(set, [(False, KnownValue(1)), (False, KnownValue(2))])
            )


class TestConstructors(TestNameCheckVisitorBase):
    @assert_passes()
    def test_init_new(self):
        def capybara():
            from _pyanalyze_tests.initnew import (
                my_enumerate,
                overloadinit,
                overloadnew,
                simple,
                simplenew,
            )

            simple()  # E: incompatible_call
            simple("x")  # E: incompatible_argument
            assert_is_value(simple(1), TypedValue("_pyanalyze_tests.initnew.simple"))

            my_enumerate()  # E: incompatible_call
            my_enumerate([1], start="x")  # E: incompatible_argument
            assert_is_value(
                my_enumerate([1]),
                GenericValue("_pyanalyze_tests.initnew.my_enumerate", [KnownValue(1)]),
            )

            overloadinit()  # E: incompatible_call
            assert_is_value(
                overloadinit(1, "x", 2),
                GenericValue("_pyanalyze_tests.initnew.overloadinit", [KnownValue(2)]),
            )

            simplenew()  # E: incompatible_call
            assert_is_value(
                simplenew(1), TypedValue("_pyanalyze_tests.initnew.simplenew")
            )

            overloadnew()  # E: incompatible_call
            assert_is_value(
                overloadnew(1, "x", 2),
                GenericValue("_pyanalyze_tests.initnew.overloadnew", [KnownValue(2)]),
            )

    @assert_passes()
    def test_typeshed_constructors(self):
        def capybara(x):
            assert_is_value(int(x), TypedValue(int))
            assert_is_value(
                frozenset(),
                GenericValue(frozenset, [AnyValue(AnySource.generic_argument)]),
            )

            assert_is_value(type("x"), TypedValue(type))
            assert_is_value(type("x", (), {}), TypedValue(type))


class Parent(Generic[T]):
    pass


class Child(Parent[int]):
    pass


class GenericChild(Parent[T]):
    pass


class TestGetGenericBases:
    def setup_method(self) -> None:
        checker = Checker()
        self.get_generic_bases = checker.arg_spec_cache.get_generic_bases

    def test_runtime(self):
        assert {
            Parent: {T: AnyValue(AnySource.generic_argument)}
        } == self.get_generic_bases(Parent)
        assert {Parent: {T: TypeVarValue(T)}} == self.get_generic_bases(
            Parent, [TypeVarValue(T)]
        )
        assert {Child: {}, Parent: {T: TypedValue(int)}} == self.get_generic_bases(
            Child
        )
        assert {
            GenericChild: {T: AnyValue(AnySource.generic_argument)},
            Parent: {T: AnyValue(AnySource.generic_argument)},
        } == self.get_generic_bases(GenericChild)
        one = KnownValue(1)
        assert {GenericChild: {T: one}, Parent: {T: one}} == self.get_generic_bases(
            GenericChild, [one]
        )

    def check(
        self,
        expected: Dict[Union[type, str], List[Value]],
        base: Union[type, str],
        args: typing.Sequence[Value] = (),
    ) -> None:
        actual = self.get_generic_bases(base, args)
        cleaned = {base: list(tv_map.values()) for base, tv_map in actual.items()}
        assert expected == cleaned

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

    def test_dict_items(self):
        TInt = TypedValue(int)
        TStr = TypedValue(str)
        TTuple = make_simple_sequence(tuple, [TInt, TStr])
        self.check(
            {
                "_collections_abc.dict_items": [TInt, TStr],
                collections.abc.Iterable: [TTuple],
                collections.abc.Container: [TTuple],
                collections.abc.Collection: [TTuple],
                collections.abc.Set: [TTuple],
                collections.abc.MappingView: [],
                collections.abc.ItemsView: [TInt, TStr],
                collections.abc.Sized: [],
            },
            "_collections_abc.dict_items",
            [TInt, TStr],
        )

    def test_struct_time(self):
        if sys.version_info < (3, 9):
            # Until 3.8 NamedTuple is actually a class.
            expected = {
                time.struct_time: [],
                "_typeshed.structseq": [AnyValue(AnySource.explicit) | TypedValue(int)],
                tuple: [TypedValue(int)],
                collections.abc.Collection: [TypedValue(int)],
                collections.abc.Reversible: [TypedValue(int)],
                collections.abc.Iterable: [TypedValue(int)],
                collections.abc.Sequence: [TypedValue(int)],
                collections.abc.Container: [TypedValue(int)],
            }
        else:
            expected = {
                time.struct_time: [],
                "_typeshed.structseq": [AnyValue(AnySource.explicit) | TypedValue(int)],
                tuple: [TypedValue(int)],
                collections.abc.Collection: [TypedValue(int)],
                collections.abc.Reversible: [TypedValue(int)],
                collections.abc.Iterable: [TypedValue(int)],
                collections.abc.Sequence: [TypedValue(int)],
                collections.abc.Container: [TypedValue(int)],
            }
        self.check(expected, time.struct_time)

    def test_context_manager(self):
        int_tv = TypedValue(int)
        missing = AnyValue(AnySource.generic_argument)
        self.check(
            {contextlib.AbstractContextManager: [int_tv, missing]},
            contextlib.AbstractContextManager,
            [int_tv],
        )
        self.check(
            {contextlib.AbstractAsyncContextManager: [int_tv, missing]},
            contextlib.AbstractAsyncContextManager,
            [int_tv],
        )

    def test_collections(self):
        int_tv = TypedValue(int)
        str_tv = TypedValue(str)
        int_str_tuple = make_simple_sequence(tuple, [int_tv, str_tv])
        self.check(
            {
                collections.abc.ValuesView: [int_tv],
                collections.abc.MappingView: [],
                collections.abc.Iterable: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Container: [int_tv],
                collections.abc.Sized: [],
            },
            collections.abc.ValuesView,
            [int_tv],
        )
        self.check(
            {
                collections.abc.ItemsView: [int_tv, str_tv],
                collections.abc.MappingView: [],
                collections.abc.Set: [int_str_tuple],
                collections.abc.Collection: [int_str_tuple],
                collections.abc.Iterable: [int_str_tuple],
                collections.abc.Container: [int_str_tuple],
                collections.abc.Sized: [],
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

    def test_parse_result(self):
        self.check(
            {
                collections.abc.Iterable: [AnyValue(AnySource.generic_argument)],
                collections.abc.Reversible: [AnyValue(AnySource.generic_argument)],
                collections.abc.Container: [AnyValue(AnySource.generic_argument)],
                collections.abc.Collection: [AnyValue(AnySource.generic_argument)],
                collections.abc.Sequence: [AnyValue(AnySource.generic_argument)],
                urllib.parse.ParseResult: [],
                urllib.parse._ParseResultBase: [],
                tuple: [AnyValue(AnySource.generic_argument)],
                urllib.parse._ResultMixinStr: [],
                urllib.parse._NetlocResultMixinBase: [TypedValue(str)],
                urllib.parse._NetlocResultMixinStr: [],
                urllib.parse._ResultMixinStr: [],
            },
            urllib.parse.ParseResult,
        )

    def test_buffered_reader(self):
        self.check(
            {
                io.IOBase: [],
                io.BufferedIOBase: [],
                collections.abc.Iterable: [TypedValue(bytes)],
                collections.abc.Iterator: [TypedValue(bytes)],
                io.BufferedReader: [],
                typing.BinaryIO: [],
                typing.IO: [TypedValue(bytes)],
            },
            io.BufferedReader,
        )


class TestAttribute:
    def test_basic(self) -> None:
        tsf = TypeshedFinder(Checker(), verbose=True)
        assert TypedValue(bool) == tsf.get_attribute(
            staticmethod, "__isabstractmethod__", on_class=False
        )

    def test_property(self) -> None:
        tsf = TypeshedFinder(Checker(), verbose=True)
        assert TypedValue(int) == tsf.get_attribute(int, "real", on_class=False)

    def test_http_error(self) -> None:
        tsf = TypeshedFinder(Checker(), verbose=True)
        assert True is tsf.has_attribute(HTTPError, "read")


class TestRange(TestNameCheckVisitorBase):
    @assert_passes()
    def test_iteration(self):
        def capybara(r: range):
            for j in r:
                assert_is_value(j, TypedValue(int))

            for i in range(10000000):
                assert_is_value(i, TypedValue(int))


class TestParamSpec(TestNameCheckVisitorBase):
    @assert_passes()
    def test_contextmanager(self):
        import contextlib
        from typing import Iterator

        def cm(a: int) -> Iterator[str]:
            yield "hello"

        def capybara():
            wrapped = contextlib.contextmanager(cm)
            assert_is_value(
                wrapped(1),
                GenericValue(contextlib._GeneratorContextManager, [TypedValue(str)]),
            )
            wrapped("x")  # E: incompatible_argument


class TestIntegration(TestNameCheckVisitorBase):
    @assert_passes()
    def test_open(self):
        import io
        from typing import IO, Any, BinaryIO

        from pyanalyze.extensions import assert_type

        def capybara(buffering: int, mode: str):
            assert_type(open("x"), io.TextIOWrapper)
            assert_type(open("x", "r"), io.TextIOWrapper)
            assert_type(open("x", "rb"), io.BufferedReader)
            assert_type(open("x", "rb", buffering=0), io.FileIO)
            assert_type(open("x", "rb", buffering=buffering), BinaryIO)
            assert_type(open("x", mode, buffering=buffering), IO[Any])

    @assert_passes()
    def test_itertools_count(self):
        import itertools

        def capybara():
            assert_is_value(
                itertools.count(1), GenericValue(itertools.count, [TypedValue(int)])
            )


class TestNestedClass(TestNameCheckVisitorBase):
    @assert_passes()
    def test_nested(self):
        def capybara() -> None:
            from _pyanalyze_tests.nested import Outer

            Outer.Inner(1)

    @assert_passes()
    def test_with_runtime_object(self):
        import sys
        import types

        class Inner:
            def __init__(self, arg: int) -> None:
                pass

        # The bug here only reproduces if a class that exists at runtime contains
        # a nested class in a stub. So we simulate that by creating a fake runtime
        # module.
        Outer = type("Outer", (), {"Inner": Inner})
        Outer.__module__ = "_pyanalyze_tests.nested"
        mod = types.ModuleType("_pyanalyze_tests.nested")
        mod.Outer = Outer
        sys.modules["_pyanalyze_tests.nested"] = mod

        def capybara() -> None:
            Outer.Inner(1)


class TestDeprecated(TestNameCheckVisitorBase):
    @assert_passes()
    def test_utcnow(self):
        import datetime

        def capybara() -> None:
            assert_is_value(datetime.datetime.utcnow(), TypedValue(datetime.datetime))
