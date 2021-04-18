# static analysis: ignore
from asynq import asynq
from qcore.asserts import assert_eq
import functools
from typing import TypeVar, NewType

from .test_name_check_visitor import (
    TestNameCheckVisitorBase,
    ConfiguredNameCheckVisitor,
)
from .test_node_visitor import assert_passes
from .signature import SigParameter, BoundMethodSignature, Signature
from .arg_spec import ArgSpecCache, is_dot_asynq_function
from .tests import l0cached_async_fn
from .value import (
    KnownValue,
    NewTypeValue,
    TypedValue,
    GenericValue,
    UNRESOLVED_VALUE,
    UNINITIALIZED_VALUE,
    SequenceIncompleteValue,
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
    visitor = ConfiguredNameCheckVisitor(__file__, "", {}, fail_after_first=False)
    config = visitor.config
    cwc_typed = TypedValue(ClassWithCall)
    cwc_self = SigParameter("self", annotation=cwc_typed)

    # test everything twice because calling qcore.get_original_fn has side effects
    for _ in range(2):

        # there's special logic for this in _get_argspec_from_value; TODO move that into
        # ExtendedArgSpec
        assert_eq(
            BoundMethodSignature(
                Signature.make(
                    [cwc_self, SigParameter("arg")], callable=ClassWithCall.__call__
                ),
                cwc_typed,
            ),
            visitor._get_argspec_from_value(cwc_typed, None),
        )

        assert_eq(
            BoundMethodSignature(
                Signature.make(
                    [SigParameter("cls")],
                    callable=ClassWithCall.normal_classmethod.__func__,
                ),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.normal_classmethod),
        )
        assert_eq(
            Signature.make(
                [SigParameter("arg")], callable=ClassWithCall.normal_staticmethod
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.normal_staticmethod),
        )

        assert_eq(
            Signature.make(
                [
                    SigParameter("capybara"),
                    SigParameter("hutia", default=KnownValue(3)),
                    SigParameter("tucotucos", SigParameter.VAR_POSITIONAL),
                    SigParameter("proechimys", SigParameter.VAR_KEYWORD),
                ],
                callable=function,
            ),
            ArgSpecCache(config).get_argspec(function),
        )

        assert_eq(
            Signature.make(
                [SigParameter("x"), SigParameter("y")], callable=async_function.fn
            ),
            ArgSpecCache(config).get_argspec(async_function),
        )

        assert_eq(
            Signature.make(
                [SigParameter("x"), SigParameter("y")], callable=async_function.fn
            ),
            ArgSpecCache(config).get_argspec(async_function.asynq),
        )

        instance = ClassWithCall(1)

        assert_eq(
            BoundMethodSignature(
                Signature.make(
                    [SigParameter("self"), SigParameter("x")],
                    callable=instance.async_method.decorator.fn,
                ),
                KnownValue(instance),
            ),
            ArgSpecCache(config).get_argspec(instance.async_method),
        )

        assert_eq(
            BoundMethodSignature(
                Signature.make(
                    [SigParameter("self"), SigParameter("x")],
                    callable=instance.async_method.decorator.fn,
                ),
                KnownValue(instance),
            ),
            ArgSpecCache(config).get_argspec(instance.async_method.asynq),
        )

        assert_eq(
            Signature.make(
                [SigParameter("y")], callable=ClassWithCall.async_staticmethod.fn
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_staticmethod),
        )

        assert_eq(
            Signature.make(
                [SigParameter("y")], callable=ClassWithCall.async_staticmethod.fn
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_staticmethod.asynq),
        )

        assert_eq(
            BoundMethodSignature(
                Signature.make(
                    [SigParameter("cls"), SigParameter("z")],
                    callable=ClassWithCall.async_classmethod.decorator.fn,
                ),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_classmethod),
        )

        assert_eq(
            BoundMethodSignature(
                Signature.make(
                    [SigParameter("cls"), SigParameter("z")],
                    callable=ClassWithCall.async_classmethod.decorator.fn,
                ),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_classmethod.asynq),
        )

        assert_eq(
            BoundMethodSignature(
                Signature.make(
                    [SigParameter("cls"), SigParameter("ac")],
                    callable=ClassWithCall.pure_async_classmethod.decorator.fn,
                ),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.pure_async_classmethod),
        )

        # This behaves differently in 3.9 (decorator) than in 3.6-3.8 (__func__). Not
        # sure why.
        if hasattr(ClassWithCall.classmethod_before_async, "decorator"):
            callable = ClassWithCall.classmethod_before_async.decorator.fn
        else:
            callable = ClassWithCall.classmethod_before_async.__func__.fn

        assert_eq(
            BoundMethodSignature(
                Signature.make(
                    [SigParameter("cls"), SigParameter("ac")], callable=callable
                ),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.classmethod_before_async),
        )

        assert_eq(
            Signature.make(
                [
                    SigParameter("args", annotation=TypedValue(int)),
                    SigParameter("kwargs", annotation=TypedValue(str)),
                ],
                KnownValue(None),
                callable=wrapped,
            ),
            ArgSpecCache(config).get_argspec(wrapped),
        )
        decorated = decorator(wrapped)
        assert_eq(
            Signature.make(
                [
                    SigParameter(
                        "args", SigParameter.VAR_POSITIONAL, annotation=UNRESOLVED_VALUE
                    ),
                    SigParameter(
                        "kwargs", SigParameter.VAR_KEYWORD, annotation=UNRESOLVED_VALUE
                    ),
                ],
                callable=decorated,
            ),
            ArgSpecCache(config).get_argspec(decorated),
        )
        assert_eq(
            Signature.make(
                [
                    SigParameter(
                        "x", SigParameter.POSITIONAL_ONLY, annotation=TypedValue(int)
                    )
                ],
                NewTypeValue(NT),
                callable=NT,
            ),
            ArgSpecCache(config).get_argspec(NT),
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
                asyncio.sleep(3), GenericValue(asyncio.Future, [UNRESOLVED_VALUE])
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
