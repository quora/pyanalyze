# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestRecursion(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import Dict, List, Union

        JSON = Union[Dict[str, "JSON"], List["JSON"], int, str, float, bool, None]

        def f(x: JSON):
            pass

        def capybara():
            f([])
            f([1, 2, 3])
            f([[{1}]])  # TODO this should throw an error


class TestTypeAliasType(TestNameCheckVisitorBase):
    @assert_passes()
    def test_typing_extensions(self):
        from typing_extensions import TypeAliasType, assert_type

        MyType = TypeAliasType("MyType", int)

        def f(x: MyType):
            assert_type(x, MyType)
            assert_type(x + 1, int)

        def capybara(i: int, s: str):
            f(i)
            f(s)  # E: incompatible_argument

    @assert_passes()
    def test_typing_extensions_generic(self):
        from typing import List, Set, TypeVar, Union

        from typing_extensions import TypeAliasType, assert_type

        T = TypeVar("T")
        MyType = TypeAliasType("MyType", Union[List[T], Set[T]], type_params=(T,))

        def f(x: MyType[int]):
            assert_type(x, MyType[int])
            assert_type(list(x), List[int])

        def capybara(i: int, s: str):
            f([i])
            f([s])  # E: incompatible_argument

    def test_312(self):
        self.assert_passes(
            """
            from typing_extensions import assert_type
            type MyType = int

            def f(x: MyType):
                assert_type(x, MyType)
                assert_type(x + 1, int)

            def capybara(i: int, s: str):
                f(i)
                f(s)  # E: incompatible_argument
        """
        )

    def test_312_generic(self):
        self.assert_passes(
            """
            from typing_extensions import assert_type
            type MyType[T] = list[T] | set[T]

            def f(x: MyType[int]):
                assert_type(x, MyType[int])
                assert_type(list(x), list[int])

            def capybara(i: int, s: str):
                f([i])
                f([s])  # E: incompatible_argument
        """
        )

    def test_312_local_alias(self):
        self.assert_passes(
            """
            from typing_extensions import assert_type

            def capybara():
                type MyType = int
                def f(x: MyType):
                    assert_type(x, MyType)
                    assert_type(x + 1, int)

                f(1)
                f("x")  # E: incompatible_argument
        """
        )

    @assert_passes()
    def test_paramspec_argument(self):
        from typing import Callable

        from typing_extensions import ParamSpec, TypeAliasType, assert_type

        P = ParamSpec("P")
        Handler = TypeAliasType("Handler", Callable[P, int], type_params=(P,))

        def capybara(handler: Handler[[int, str]]) -> None:
            assert_type(handler(1, "x"), int)
            handler("x")  # E: incompatible_call

    @assert_passes()
    def test_call(self):
        from typing import Callable

        from typing_extensions import TypeAliasType, assert_type

        Callback = TypeAliasType("Callback", Callable[[int], str])

        def capybara(callback: Callback) -> None:
            assert_type(callback(1), str)
            callback("x")  # E: incompatible_argument

    def test_312_truthiness(self):
        self.assert_passes(
            """
            from typing_extensions import assert_type
            type MaybeInt = int | None

            def capybara(x: MaybeInt) -> None:
                if x:
                    assert_type(x, MaybeInt)
                assert_type(not x, bool)
        """
        )
