# static analysis: ignore
from asynq import asynq
import functools
from typing import TypeVar, NewType

from .checker import Checker
from .test_name_check_visitor import (
    TestNameCheckVisitorBase,
    ConfiguredNameCheckVisitor,
)
from .test_node_visitor import assert_passes
from .signature import SigParameter, BoundMethodSignature, Signature, ParameterKind
from .stacked_scopes import Composite
from .arg_spec import ArgSpecCache, is_dot_asynq_function
from .tests import l0cached_async_fn
from .value import (
    AnySource,
    AnyValue,
    KnownValue,
    MultiValuedValue,
    NewTypeValue,
    TypedValue,
    GenericValue,
    SequenceIncompleteValue,
    assert_is_value,
)

T = TypeVar("T")
NT = NewType("NT", int)


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


def wrapped(args: int, kwargs: str) -> None:
    pass


def decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def test_get_argspec():
    config = ConfiguredNameCheckVisitor.config
    checker = Checker(config)
    visitor = ConfiguredNameCheckVisitor(
        __file__, "", {}, fail_after_first=False, checker=checker
    )
    cwc_typed = TypedValue(ClassWithCall)

    # test everything twice because calling qcore.get_original_fn has side effects
    for _ in range(2):
        asc = ArgSpecCache(checker.options, checker.ts_finder)

        # there's special logic for this in signature_from_value; TODO move that into
        # ExtendedArgSpec
        assert Signature.make(
            [SigParameter("arg")], callable=ClassWithCall.__call__
        ) == visitor.signature_from_value(cwc_typed)

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("cls")],
                callable=ClassWithCall.normal_classmethod.__func__,
            ),
            Composite(KnownValue(ClassWithCall)),
        ) == asc.get_argspec(ClassWithCall.normal_classmethod)
        assert Signature.make(
            [SigParameter("arg")], callable=ClassWithCall.normal_staticmethod
        ) == asc.get_argspec(ClassWithCall.normal_staticmethod)

        assert Signature.make(
            [
                SigParameter("capybara"),
                SigParameter("hutia", default=KnownValue(3)),
                SigParameter("tucotucos", ParameterKind.VAR_POSITIONAL),
                SigParameter("proechimys", ParameterKind.VAR_KEYWORD),
            ],
            callable=function,
        ) == asc.get_argspec(function)

        assert Signature.make(
            [SigParameter("x"), SigParameter("y")],
            callable=async_function.fn,
            is_asynq=True,
        ) == asc.get_argspec(async_function)

        assert Signature.make(
            [SigParameter("x"), SigParameter("y")],
            callable=async_function.fn,
            is_asynq=True,
        ) == asc.get_argspec(async_function.asynq)

        instance = ClassWithCall(1)

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("self"), SigParameter("x")],
                callable=instance.async_method.decorator.fn,
                is_asynq=True,
            ),
            Composite(KnownValue(instance)),
        ) == asc.get_argspec(instance.async_method)

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("self"), SigParameter("x")],
                callable=instance.async_method.decorator.fn,
                is_asynq=True,
            ),
            Composite(KnownValue(instance)),
        ) == asc.get_argspec(instance.async_method.asynq)

        assert Signature.make(
            [SigParameter("y")],
            callable=ClassWithCall.async_staticmethod.fn,
            is_asynq=True,
        ) == asc.get_argspec(ClassWithCall.async_staticmethod)

        assert Signature.make(
            [SigParameter("y")],
            callable=ClassWithCall.async_staticmethod.fn,
            is_asynq=True,
        ) == asc.get_argspec(ClassWithCall.async_staticmethod.asynq)

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("cls"), SigParameter("z")],
                callable=ClassWithCall.async_classmethod.decorator.fn,
                is_asynq=True,
            ),
            Composite(KnownValue(ClassWithCall)),
        ) == asc.get_argspec(ClassWithCall.async_classmethod)

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("cls"), SigParameter("z")],
                callable=ClassWithCall.async_classmethod.decorator.fn,
                is_asynq=True,
            ),
            Composite(KnownValue(ClassWithCall)),
        ) == asc.get_argspec(ClassWithCall.async_classmethod.asynq)

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("cls"), SigParameter("ac")],
                callable=ClassWithCall.pure_async_classmethod.decorator.fn,
            ),
            Composite(KnownValue(ClassWithCall)),
        ) == asc.get_argspec(ClassWithCall.pure_async_classmethod)

        # This behaves differently in 3.9 (decorator) than in 3.6-3.8 (__func__). Not
        # sure why.
        if hasattr(ClassWithCall.classmethod_before_async, "decorator"):
            callable = ClassWithCall.classmethod_before_async.decorator.fn
        else:
            callable = ClassWithCall.classmethod_before_async.__func__.fn

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("cls"), SigParameter("ac")],
                callable=callable,
                is_asynq=True,
            ),
            Composite(KnownValue(ClassWithCall)),
        ) == asc.get_argspec(ClassWithCall.classmethod_before_async)

        assert Signature.make(
            [
                SigParameter("args", annotation=TypedValue(int)),
                SigParameter("kwargs", annotation=TypedValue(str)),
            ],
            KnownValue(None),
            callable=wrapped,
        ) == asc.get_argspec(wrapped)
        decorated = decorator(wrapped)
        assert Signature.make(
            [
                SigParameter(
                    "args",
                    ParameterKind.VAR_POSITIONAL,
                    annotation=AnyValue(AnySource.inference),
                ),
                SigParameter(
                    "kwargs",
                    ParameterKind.VAR_KEYWORD,
                    annotation=AnyValue(AnySource.inference),
                ),
            ],
            callable=decorated,
        ) == asc.get_argspec(decorated)
        assert Signature.make(
            [
                SigParameter(
                    "x", ParameterKind.POSITIONAL_ONLY, annotation=TypedValue(int)
                )
            ],
            NewTypeValue(NT),
            callable=NT,
        ) == asc.get_argspec(NT)


def test_positional_only():
    def f(__x, _f__x):
        pass

    class Y:
        def f(self, __x):
            pass

        class X:
            def f(self, __x, _Y__x):
                pass

    asc = Checker(ConfiguredNameCheckVisitor.config).arg_spec_cache
    assert asc.get_argspec(f) == Signature.make(
        [
            SigParameter("__x", ParameterKind.POSITIONAL_ONLY),
            SigParameter("_f__x", ParameterKind.POSITIONAL_OR_KEYWORD),
        ],
        callable=f,
    )
    assert asc.get_argspec(Y.f) == Signature.make(
        [
            SigParameter("self", ParameterKind.POSITIONAL_ONLY),
            SigParameter("_Y__x", ParameterKind.POSITIONAL_ONLY),
        ],
        callable=Y.f,
    )
    assert asc.get_argspec(Y.X.f) == Signature.make(
        [
            SigParameter("self", ParameterKind.POSITIONAL_ONLY),
            SigParameter("_X__x", ParameterKind.POSITIONAL_ONLY),
            SigParameter("_Y__x", ParameterKind.POSITIONAL_OR_KEYWORD),
        ],
        callable=Y.X.f,
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
                asyncio.sleep(3),
                GenericValue(asyncio.Future, [AnyValue(AnySource.unannotated)]),
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


class TestClassInstantiation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_union_with_impl(self):
        def capybara(cond: bool) -> None:
            if cond:
                typ = list
            else:
                typ = tuple
            assert_is_value(typ, KnownValue(list) | KnownValue(tuple))
            assert_is_value(typ([1]), KnownValue([1]) | KnownValue((1,)))

    @assert_passes()
    def test_union_without_impl(self):
        class A:
            pass

        class B:
            pass

        def capybara(cond: bool) -> None:
            if cond:
                cls = A
            else:
                cls = B
            assert_is_value(cls(), MultiValuedValue([TypedValue(A), TypedValue(B)]))

    @assert_passes()
    def test_constructor_impl(self):
        from pyanalyze.tests import FailingImpl

        def capybara():
            FailingImpl()  # E: incompatible_call

    @assert_passes()
    def test_subclass_value(self):
        from typing import Type

        class A:
            def __init__(self, x: int) -> None:
                pass

        def capybara(t: Type[A]) -> None:
            assert_is_value(t(1), TypedValue(A))
            t("x")  # E: incompatible_argument


class TestFunctionsSafeToCall(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def test(self):
            assert_is_value(sorted([3, 1, 2]), KnownValue([1, 2, 3]))


class TestNamedTuple(TestNameCheckVisitorBase):
    @assert_passes()
    def test_args(self):
        from typing import NamedTuple

        class NT(NamedTuple):
            field: int

        class CustomNew:
            def __new__(self, a: int) -> "CustomNew":
                return super().__new__(self)

        def make_nt() -> NT:
            return NT(field=3)

        def capybara():
            NT(filed=3)  # E: incompatible_call
            nt2 = make_nt()
            assert_is_value(nt2, TypedValue(NT))
            assert_is_value(nt2.field, TypedValue(int))

            CustomNew("x")  # E: incompatible_argument
            cn = CustomNew(a=3)
            assert_is_value(cn, TypedValue(CustomNew))


class TestBuiltinMethods(TestNameCheckVisitorBase):
    @assert_passes()
    def test_method_wrapper(self):
        import collections.abc

        def capybara():
            r = range(10)
            assert_is_value(r, KnownValue(range(10)))
            assert_is_value(
                r.__iter__(), GenericValue(collections.abc.Iterator, [TypedValue(int)])
            )
