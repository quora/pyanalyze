# static analysis: ignore
from .value import GenericValue, KnownValue, AnyValue, AnySource, TypedValue
from .error_code import ErrorCode
from .implementation import assert_is_value
from .test_node_visitor import assert_passes, skip_before
from .test_name_check_visitor import TestNameCheckVisitorBase


class TestNestedFunction(TestNameCheckVisitorBase):
    @assert_passes()
    def test_inference(self):
        def capybara():
            def nested():
                pass

            class NestedClass(object):
                pass

            assert_is_value(nested(), KnownValue(None))
            nested(1)  # E: incompatible_call
            # Should ideally be something more specific
            assert_is_value(NestedClass, AnyValue(AnySource.inference))

    @assert_passes()
    def test_usage_in_nested_scope():
        def capybara(cond, x):
            if cond:

                def nested(y):
                    pass

                ys = [nested(y) for y in x]

                class Nested(object):
                    xs = ys

    @assert_passes()
    def test_asynq(self):
        from asynq import asynq
        from typing_extensions import Literal

        @asynq()
        def capybara():
            @asynq()
            def nested() -> Literal[3]:
                return 3

            assert_is_value(nested(), KnownValue(3))
            val = yield nested.asynq()
            assert_is_value(val, KnownValue(3))

    @assert_passes()
    def test_async_def(self):
        import collections.abc

        def capybara():
            async def nested() -> int:
                return 1

            assert_is_value(
                nested(), GenericValue(collections.abc.Awaitable, [TypedValue(int)])
            )

    @assert_passes()
    def test_bad_decorator(self):
        def decorator(fn):
            return fn

        def capybara():
            @decorator
            def nested():
                pass

            assert_is_value(nested, AnyValue(AnySource.unannotated))

    @assert_passes()
    def test_attribute_set(self):
        def capybara():
            def inner():
                pass

            inner.punare = 3
            assert_is_value(inner.punare, KnownValue(3))

    @assert_passes()
    def test_nested_in_method(self):
        class Capybara:
            def method(self):
                def nested(arg) -> int:
                    assert_is_value(arg, AnyValue(AnySource.unannotated))
                    # Make sure we don't think this is an access to Capybara.numerator
                    print(arg.numerator)
                    return 1

                assert_is_value(nested(1), TypedValue(int))


class TestFunctionDefinitions(TestNameCheckVisitorBase):
    @assert_passes()
    def test_keyword_only(self):
        def capybara(a, *, b, c=3):
            assert_is_value(a, AnyValue(AnySource.unannotated))
            assert_is_value(b, AnyValue(AnySource.unannotated))
            assert_is_value(c, AnyValue(AnySource.unannotated) | KnownValue(3))
            capybara(1, b=2)

            fn = lambda a, *, b: None
            fn(a, b=b)

        def failing_capybara(a, *, b):
            capybara(1, 2)  # E: incompatible_call

    @skip_before((3, 8))
    def test_pos_only(self):
        self.assert_passes(
            """
            from typing import Optional

            def f(a: int, /) -> None:
                assert_is_value(a, TypedValue(int))

            def g(a: Optional[str] = None, /, b: int = 1) -> None:
                assert_is_value(a, TypedValue(str) | KnownValue(None))
                assert_is_value(b, TypedValue(int))

            def h(a, b: int = 1, /, c: int = 2) -> None:  # E: missing_parameter_annotation
                assert_is_value(a, AnyValue(AnySource.unannotated))
                assert_is_value(b, TypedValue(int))

            def capybara() -> None:
                f(1)
                f("x")  # E: incompatible_argument
                f(a=1)  # E: incompatible_call
                g(a=1)  # E: incompatible_call
                g(b=1)
                g(None, b=1)
                h(1, 1, c=2)
                h(1)
                h(1, b=1)  # E: incompatible_call
            """,
            settings={ErrorCode.missing_parameter_annotation: True},
        )

    @assert_passes()
    def test_lambda(self):
        from typing import Callable

        def capybara():
            fun = lambda: 1
            x: Callable[[], int] = fun
            y: Callable[[], str] = fun  # E: incompatible_assignment
            fun(1)  # E: incompatible_call
            assert_is_value(fun(), KnownValue(1))

            fun2 = lambda a: a
            fun2()  # E: incompatible_call
            assert_is_value(fun2(1), KnownValue(1))

            fun3 = lambda c=3: c
            assert_is_value(
                fun3(), KnownValue(3) | AnyValue(AnySource.generic_argument)
            )
            assert_is_value(fun3(2), KnownValue(2) | KnownValue(3))

            fun4 = lambda a, b, c: a if c else b
            assert_is_value(fun4(1, 2, 3), KnownValue(1) | KnownValue(2))
