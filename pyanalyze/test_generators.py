# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import AnySource, AnyValue, KnownValue, TypedValue


class TestGenerator(TestNameCheckVisitorBase):
    @assert_passes()
    def test_generator_return(self):
        from typing import Generator

        def gen(cond) -> Generator[int, str, float]:
            x = yield 1
            assert_is_value(x, TypedValue(str))
            yield "x"  # E: incompatible_yield
            if cond:
                return 3.0
            else:
                return "hello"  # E: incompatible_return_value

        def capybara() -> Generator[int, int, int]:
            x = yield from gen(True)  # E: incompatible_yield
            assert_is_value(x, TypedValue(float))

            return 3

    @assert_passes()
    def test_iterable_return(self):
        from typing import Iterable

        def gen(cond) -> Iterable[int]:
            x = yield 1
            assert_is_value(x, KnownValue(None))

            yield "x"  # E: incompatible_yield

            if cond:
                return
            else:
                return 3  # E: incompatible_return_value

        def caller() -> Iterable[int]:
            x = yield from gen(True)
            assert_is_value(x, AnyValue(AnySource.generic_argument))


class TestGeneratorReturn(TestNameCheckVisitorBase):
    @assert_passes()
    def test_sync(self):
        from typing import Generator, Iterable

        from asynq import ConstFuture, asynq

        def gen() -> int:  # E: generator_return
            yield 1

        def caller() -> int:  # E: generator_return
            x = yield from [1, 2]
            print(x)

        def gen2() -> Iterable[int]:
            yield 1

        def caller2() -> Generator[int, None, None]:
            x = yield from [1, 2]
            print(x)

        @asynq()
        def asynq_gen() -> int:
            x = yield ConstFuture(1)
            return x

    @assert_passes()
    def test_async(self):
        from typing import AsyncGenerator, AsyncIterable

        async def gen() -> int:  # E: generator_return
            yield 1

        async def gen2() -> AsyncIterable[int]:
            yield 1

        async def gen3() -> AsyncGenerator[int, None]:
            yield 1
