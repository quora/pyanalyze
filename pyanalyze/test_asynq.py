# static analysis: ignore
from .error_code import ErrorCode
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_fails, assert_passes
from .tests import make_simple_sequence
from .value import (
    AnySource,
    AnyValue,
    AsyncTaskIncompleteValue,
    DictIncompleteValue,
    KnownValue,
    KVPair,
    TypedValue,
)


class TestBadAsyncYield(TestNameCheckVisitorBase):
    @assert_passes()
    def test_const_future(self):
        from asynq import asynq, ConstFuture, FutureBase

        @asynq()
        def capybara(condition):
            yield FutureBase()
            val = yield ConstFuture(3)
            assert_is_value(val, KnownValue(3))
            val2 = yield None
            assert_is_value(val2, KnownValue(None))

            if condition:
                task = ConstFuture(4)
            else:
                task = capybara.asynq(True)
            val3 = yield task
            assert_is_value(val3, KnownValue(4) | AnyValue(AnySource.inference))


class TestUnwrapYield(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import Sequence

        from asynq import asynq
        from typing_extensions import Literal

        @asynq()
        def async_fn(n):
            return "async_fn"

        @asynq()
        def square(n):
            return int(n * n)

        class Capybara(object):
            @asynq()
            def async_method(self):
                return "capybara"

        @asynq()
        def caller(ints: Sequence[Literal[0, 1, 2]]):
            val1 = yield async_fn.asynq(1)
            assert_is_value(val1, KnownValue("async_fn"))
            val2 = yield square.asynq(3)
            assert_is_value(val2, TypedValue(int))

            val3, val4 = yield async_fn.asynq(1), async_fn.asynq(2)
            assert_is_value(val3, KnownValue("async_fn"))
            assert_is_value(val4, KnownValue("async_fn"))

            val5 = yield Capybara().async_method.asynq()
            assert_is_value(val5, KnownValue("capybara"))

            vals1 = yield [square.asynq(1), square.asynq(2), square.asynq(3)]
            assert_is_value(
                vals1,
                make_simple_sequence(
                    list, [TypedValue(int), TypedValue(int), TypedValue(int)]
                ),
            )

            vals2 = yield [square.asynq(i) for i in ints]
            for val in vals2:
                assert_is_value(val, TypedValue(int))

            vals3 = yield {1: square.asynq(1)}
            assert_is_value(
                vals3,
                DictIncompleteValue(dict, [KVPair(KnownValue(1), TypedValue(int))]),
            )

            vals4 = yield {i: square.asynq(i) for i in ints}
            assert_is_value(
                vals4,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(
                            KnownValue(0) | KnownValue(1) | KnownValue(2),
                            TypedValue(int),
                            is_many=True,
                        )
                    ],
                ),
            )


class TestTaskNeedsYield(TestNameCheckVisitorBase):
    # couldn't change assert_fails to assert_passes for
    # constfuture, async, and yielded because changes between Python 3.7 and 3.8
    @assert_fails(ErrorCode.task_needs_yield)
    def test_constfuture(self):
        from asynq import asynq, ConstFuture

        @asynq()
        def bad_async_fn():
            return ConstFuture(3)

    @assert_fails(ErrorCode.task_needs_yield)
    def test_async(self):
        from asynq import asynq

        @asynq()
        def async_fn():
            pass

        @asynq()
        def bad_async_fn():
            return async_fn.asynq()

    @assert_fails(ErrorCode.task_needs_yield)
    def test_not_yielded(self):
        from asynq import asynq

        from pyanalyze.tests import async_fn

        @asynq()
        def capybara(oid):
            return async_fn.asynq(oid)

    def test_not_yielded_replacement(self):
        self.assert_is_changed(
            """
            from asynq import asynq
            from pyanalyze.tests import async_fn

            @asynq()
            def capybara(oid):
                async_fn.asynq(oid)
            """,
            """
            from asynq import asynq
            from pyanalyze.tests import async_fn

            @asynq()
            def capybara(oid):
                yield async_fn.asynq(oid)
            """,
        )


class TestReturn(TestNameCheckVisitorBase):
    @assert_passes()
    def test_type_inference(self):
        from asynq import async_proxy, AsyncTask, asynq, ConstFuture, FutureBase

        def returns_3():
            return 3

        @asynq(pure=True)
        def pure_async_fn():
            return 4

        @asynq()
        def async_fn():
            return 3

        class WithAProperty(object):
            @property
            def this_is_one(self):
                return str(5)

        @async_proxy(pure=True)
        def pure_async_proxy(oid):
            return ConstFuture(oid)

        @async_proxy()
        def impure_async_proxy():
            return ConstFuture(3)

        def capybara(oid):
            assert_is_value(returns_3(), KnownValue(3))
            assert_is_value(
                pure_async_fn(), AsyncTaskIncompleteValue(AsyncTask, KnownValue(4))
            )
            assert_is_value(async_fn(), KnownValue(3))
            assert_is_value(
                async_fn.asynq(), AsyncTaskIncompleteValue(AsyncTask, KnownValue(3))
            )
            assert_is_value(WithAProperty().this_is_one, TypedValue(str))
            assert_is_value(pure_async_proxy(oid), AnyValue(AnySource.unannotated))
            assert_is_value(impure_async_proxy(), AnyValue(AnySource.unannotated))
            assert_is_value(
                impure_async_proxy.asynq(),
                AsyncTaskIncompleteValue(FutureBase, AnyValue(AnySource.unannotated)),
            )

    # Can't use assert_passes for those two because the location of the error
    # changes between 3.7 and 3.8. Maybe we should hack the error code to
    # always show the error for a function on the def line, not the decorator line.
    @assert_fails(ErrorCode.missing_return)
    def test_asynq_missing_return(self):
        from asynq import asynq

        @asynq()  # E: missing_return
        def f() -> int:
            yield f.asynq()

    @assert_fails(ErrorCode.missing_return)
    def test_asynq_missing_branch(self):
        from asynq import asynq

        @asynq()  # E: missing_return
        def capybara(cond: bool) -> int:
            if cond:
                return 3
            yield capybara.asynq(False)
