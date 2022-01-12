# static analysis: ignore
from qcore.testing import Anything
import collections.abc
from collections.abc import MutableSequence, Sequence, Collection, Reversible, Set
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import textwrap
import time
from typeshed_client import Resolver, get_search_context
import typing
from typing import Dict, Generic, List, TypeVar, NewType, Union
from urllib.error import HTTPError
import urllib.parse

from .extensions import evaluated
from .test_config import TestConfig, TEST_OPTIONS
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .options import Options
from .signature import SigParameter, Signature
from .arg_spec import ArgSpecCache
from .test_arg_spec import ClassWithCall
from .typeshed import TypeshedFinder
from .value import (
    SubclassValue,
    TypedDictValue,
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
        tsf = TypeshedFinder(verbose=True)
        assert [
            GenericValue(MutableSequence, (TypeVarValue(typevar=Anything),)),
            GenericValue(Generic, (TypeVarValue(typevar=Anything),)),
        ] == tsf.get_bases(list)
        assert [
            GenericValue(Collection, (TypeVarValue(typevar=Anything),)),
            GenericValue(Reversible, (TypeVarValue(typevar=Anything),)),
            GenericValue(Generic, (TypeVarValue(typevar=Anything),)),
        ] == tsf.get_bases(Sequence)
        assert [GenericValue(Collection, (TypeVarValue(Anything),))] == tsf.get_bases(
            Set
        )

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
            tsf = TypeshedFinder(verbose=True)
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
                make_weak(GenericValue(dict, [TypedValue(int), TypedValue(str)])),
            )

    @assert_passes()
    def test_str_find(self):
        def capybara(s: str) -> None:
            assert_is_value(s.find("x"), TypedValue(int))

    @assert_passes()
    def test_str_count(self):
        def capybara(s: str) -> None:
            assert_is_value(s.count("x"), TypedValue(int))

    def test_has_stubs(self) -> None:
        tsf = TypeshedFinder(verbose=True)
        assert tsf.has_stubs(object)
        assert not tsf.has_stubs(ClassWithCall)

    def test_get_attribute(self) -> None:
        tsf = TypeshedFinder(verbose=True)
        assert UNINITIALIZED_VALUE is tsf.get_attribute(object, "nope", on_class=False)
        assert TypedValue(bool) == tsf.get_attribute(
            staticmethod, "__isabstractmethod__", on_class=False
        )


_EXPECTED_TYPED_DICTS = {
    "TD1": TypedDictValue({"a": (True, TypedValue(int)), "b": (True, TypedValue(str))}),
    "TD2": TypedDictValue(
        {"a": (False, TypedValue(int)), "b": (False, TypedValue(str))}
    ),
    "PEP655": TypedDictValue(
        {"a": (False, TypedValue(int)), "b": (True, TypedValue(str))}
    ),
    "Inherited": TypedDictValue(
        {
            "a": (True, TypedValue(int)),
            "b": (True, TypedValue(str)),
            "c": (True, TypedValue(float)),
        }
    ),
}


class TestBundledStubs(TestNameCheckVisitorBase):
    @assert_passes()
    def test_import_aliases(self):
        def capybara():
            from _pyanalyze_tests.aliases import (
                constant,
                aliased_constant,
                explicitly_aliased_constant,
                ExplicitAlias,
            )

            assert_is_value(ExplicitAlias, KnownValue(int))
            assert_is_value(constant, TypedValue(int))
            assert_is_value(aliased_constant, TypedValue(int))
            assert_is_value(explicitly_aliased_constant, TypedValue(int))

    def test_aliases(self):
        tsf = TypeshedFinder.make(TEST_OPTIONS, verbose=True)
        mod = "_pyanalyze_tests.aliases"
        assert tsf.resolve_name(mod, "constant") == TypedValue(int)
        assert tsf.resolve_name(mod, "aliased_constant") == TypedValue(int)
        assert tsf.resolve_name(mod, "explicitly_aliased_constant") == TypedValue(int)

    def test_typeddict(self):
        tsf = TypeshedFinder.make(TEST_OPTIONS, verbose=True)
        mod = "_pyanalyze_tests.typeddict"

        for name, expected in _EXPECTED_TYPED_DICTS.items():
            assert tsf.resolve_name(mod, name) == SubclassValue(expected, exactly=True)

    @assert_passes()
    def test_import_typeddicts(self):
        def capybara():
            from _pyanalyze_tests.typeddict import TD1, TD2, PEP655, Inherited
            from pyanalyze.test_typeshed import _EXPECTED_TYPED_DICTS

            def nested(td1: TD1, td2: TD2, pep655: PEP655, inherited: Inherited):
                assert_is_value(td1, _EXPECTED_TYPED_DICTS["TD1"])
                assert_is_value(td2, _EXPECTED_TYPED_DICTS["TD2"])
                assert_is_value(pep655, _EXPECTED_TYPED_DICTS["PEP655"])
                assert_is_value(inherited, _EXPECTED_TYPED_DICTS["Inherited"])

    def test_evaluated(self):
        tsf = TypeshedFinder.make(TEST_OPTIONS, verbose=True)
        mod = "_pyanalyze_tests.evaluated"
        assert tsf.resolve_name(mod, "evaluated") == KnownValue(evaluated)

    @assert_passes()
    def test_evaluated_import(self):
        def capybara(unannotated):
            from _pyanalyze_tests.evaluated import open, open2
            from typing import TextIO, BinaryIO, IO

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


class Parent(Generic[T]):
    pass


class Child(Parent[int]):
    pass


class GenericChild(Parent[T]):
    pass


class TestGetGenericBases:
    def setup(self) -> None:
        arg_spec_cache = ArgSpecCache(
            Options.from_option_list([], TestConfig()), TypeshedFinder()
        )
        self.get_generic_bases = arg_spec_cache.get_generic_bases

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
        TTuple = SequenceIncompleteValue(tuple, [TInt, TStr])
        self.check(
            {
                "_collections_abc.dict_items": [TInt, TStr],
                collections.abc.Iterable: [TTuple],
                collections.abc.Sized: [],
                collections.abc.Container: [TTuple],
                collections.abc.Collection: [TTuple],
                collections.abc.Set: [TTuple],
                collections.abc.MappingView: [],
                collections.abc.ItemsView: [TInt, TStr],
            },
            "_collections_abc.dict_items",
            [TInt, TStr],
        )

    def test_struct_time(self):
        if sys.version_info < (3, 9):
            # Until 3.8 NamedTuple is actually a class.
            expected = {
                time.struct_time: [],
                "time._struct_time": [],
                typing.NamedTuple: [],
                # Ideally should be not Any, but we haven't implemented
                # support for typeshed namedtuples.
                tuple: [AnyValue(AnySource.explicit)],
                collections.abc.Collection: [AnyValue(AnySource.explicit)],
                collections.abc.Reversible: [AnyValue(AnySource.explicit)],
                collections.abc.Iterable: [AnyValue(AnySource.explicit)],
                collections.abc.Sequence: [AnyValue(AnySource.explicit)],
                collections.abc.Container: [AnyValue(AnySource.explicit)],
            }
        else:
            expected = {
                time.struct_time: [],
                "time._struct_time": [],
                # Ideally should be not Any, but we haven't implemented
                # support for typeshed namedtuples.
                tuple: [AnyValue(AnySource.generic_argument)],
                collections.abc.Collection: [AnyValue(AnySource.generic_argument)],
                collections.abc.Reversible: [AnyValue(AnySource.generic_argument)],
                collections.abc.Iterable: [AnyValue(AnySource.generic_argument)],
                collections.abc.Sequence: [AnyValue(AnySource.generic_argument)],
                collections.abc.Container: [AnyValue(AnySource.generic_argument)],
            }
        self.check(expected, time.struct_time)

    def test_context_manager(self):
        int_tv = TypedValue(int)
        self.check(
            {contextlib.AbstractContextManager: [int_tv]},
            contextlib.AbstractContextManager,
            [int_tv],
        )
        if sys.version_info >= (3, 7):
            self.check(
                {contextlib.AbstractAsyncContextManager: [int_tv]},
                contextlib.AbstractAsyncContextManager,
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
                "urllib.parse._ResultMixinBase": [TypedValue(str)],
                tuple: [AnyValue(AnySource.generic_argument)],
                urllib.parse._ResultMixinStr: [],
                urllib.parse._NetlocResultMixinBase: [TypedValue(str)],
                urllib.parse._NetlocResultMixinStr: [],
                urllib.parse._ResultMixinStr: [],
            },
            urllib.parse.ParseResult,
        )


class TestAttribute:
    def test_basic(self) -> None:
        tsf = TypeshedFinder(verbose=True)
        assert TypedValue(bool) == tsf.get_attribute(
            staticmethod, "__isabstractmethod__", on_class=False
        )

    def test_property(self) -> None:
        tsf = TypeshedFinder(verbose=True)
        assert TypedValue(int) == tsf.get_attribute(int, "real", on_class=False)

    def test_http_error(self) -> None:
        tsf = TypeshedFinder(verbose=True)
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
