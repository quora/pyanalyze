# static analysis: ignore
from asynq import asynq
from qcore.asserts import assert_eq, assert_is
from qcore.testing import Anything
import collections.abc
from collections.abc import (
    MutableSequence,
    Sequence,
    Collection,
    Reversible,
    Set,
)
import contextlib
import io
import itertools
import time
import typing
from typing import Generic, TypeVar

from .test_config import TestConfig
from .test_name_check_visitor import (
    TestNameCheckVisitorBase,
    ConfiguredNameCheckVisitor,
)
from .test_node_visitor import assert_passes
from .arg_spec import (
    ArgSpecCache,
    BoundMethodArgSpecWrapper,
    ExtendedArgSpec,
    Parameter,
    TypeshedFinder,
    is_dot_asynq_function,
)
from .tests import l0cached_async_fn
from .value import (
    KnownValue,
    TypedValue,
    GenericValue,
    TypeVarValue,
    UNRESOLVED_VALUE,
    SequenceIncompleteValue,
)

T = TypeVar("T")


class ClassWithCall(object):
    def __init__(self, name):
        pass

    def __call__(self, arg):
        pass

    @classmethod
    def normal_classmethod(cls):
        pass

    @staticmethod
    def normal_staticmethod(arg):
        pass

    @asynq()
    def async_method(self, x):
        pass

    @asynq()
    @staticmethod
    def async_staticmethod(y):
        pass

    @asynq()
    @classmethod
    def async_classmethod(cls, z):
        pass

    @asynq(pure=True)
    @classmethod
    def pure_async_classmethod(cls, ac):
        pass

    @classmethod
    @asynq()
    def classmethod_before_async(cls, ac):
        pass


def function(capybara, hutia=3, *tucotucos, **proechimys):
    pass


@asynq()
def async_function(x, y):
    pass


def test_get_argspec():
    visitor = ConfiguredNameCheckVisitor(__file__, u"", {}, fail_after_first=False)
    config = visitor.config
    cwc_typed = TypedValue(ClassWithCall)
    cwc_self = Parameter("self", typ=cwc_typed)

    # test everything twice because calling qcore.get_original_fn has side effects
    for _ in range(2):

        # there's special logic for this in _get_argspec_from_value; TODO move that into
        # ExtendedArgSpec
        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec(
                    arguments=[cwc_self, Parameter("arg")], name="ClassWithCall"
                ),
                cwc_typed,
            ),
            visitor._get_argspec_from_value(cwc_typed, None),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("cls")]), KnownValue(ClassWithCall)
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.normal_classmethod),
        )
        assert_eq(
            ExtendedArgSpec([Parameter("arg")]),
            ArgSpecCache(config).get_argspec(ClassWithCall.normal_staticmethod),
        )

        assert_eq(
            ExtendedArgSpec(
                [Parameter("capybara"), Parameter("hutia", default_value=3)],
                starargs="tucotucos",
                kwargs="proechimys",
            ),
            ArgSpecCache(config).get_argspec(function),
        )

        assert_eq(
            ExtendedArgSpec([Parameter("x"), Parameter("y")]),
            ArgSpecCache(config).get_argspec(async_function),
        )

        assert_eq(
            ExtendedArgSpec([Parameter("x"), Parameter("y")]),
            ArgSpecCache(config).get_argspec(async_function.asynq),
        )

        instance = ClassWithCall(1)

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("self"), Parameter("x")]),
                KnownValue(instance),
            ),
            ArgSpecCache(config).get_argspec(instance.async_method),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("self"), Parameter("x")]),
                KnownValue(instance),
            ),
            ArgSpecCache(config).get_argspec(instance.async_method.asynq),
        )

        assert_eq(
            ExtendedArgSpec([Parameter("y")]),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_staticmethod),
        )

        assert_eq(
            ExtendedArgSpec([Parameter("y")]),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_staticmethod.asynq),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("cls"), Parameter("z")]),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_classmethod),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("cls"), Parameter("z")]),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_classmethod.asynq),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("cls"), Parameter("ac")]),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.pure_async_classmethod),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("cls"), Parameter("ac")]),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.classmethod_before_async),
        )


def test_is_dot_asynq_function():
    assert not is_dot_asynq_function(async_function)
    assert is_dot_asynq_function(async_function.asynq)
    assert not is_dot_asynq_function(l0cached_async_fn)
    assert is_dot_asynq_function(l0cached_async_fn.asynq)
    assert not is_dot_asynq_function(l0cached_async_fn.dirty)


class TestCoroutines(TestNameCheckVisitorBase):
    @assert_passes()
    def test_asyncio_coroutine(self):
        import asyncio
        from collections.abc import Awaitable

        @asyncio.coroutine
        def f():
            yield from asyncio.sleep(3)
            return 42

        @asyncio.coroutine
        def g():
            assert_is_value(f(), GenericValue(Awaitable, [KnownValue(42)]))

    @assert_passes()
    def test_coroutine_from_typeshed(self):
        import asyncio

        async def capybara():
            # annotated as def ... -> Future in typeshed
            assert_is_value(
                asyncio.sleep(3), GenericValue(asyncio.Future, [KnownValue(None)])
            )
            return 42

    @assert_passes()
    def test_async_def_from_typeshed(self):
        from asyncio.streams import open_connection, StreamReader, StreamWriter
        from collections.abc import Awaitable

        async def capybara():
            # annotated as async def in typeshed
            assert_is_value(
                open_connection(),
                GenericValue(
                    Awaitable,
                    [
                        SequenceIncompleteValue(
                            tuple, [TypedValue(StreamReader), TypedValue(StreamWriter)]
                        )
                    ],
                ),
            )
            return 42

    @assert_passes()
    def test_async_def(self):
        from collections.abc import Awaitable

        async def f():
            return 42

        async def g():
            assert_is_value(f(), GenericValue(Awaitable, [KnownValue(42)]))


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

    @assert_passes()
    def test_generic_self(self):
        from typing import Dict

        def capybara(x: Dict[int, str]):
            assert_is_value(
                {k: v for k, v in x.items()},
                GenericValue(dict, [TypedValue(int), TypedValue(str)]),
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
