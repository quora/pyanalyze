# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestUnsafeOverlap(TestNameCheckVisitorBase):
    @assert_passes()
    def test_simple(self):
        from typing_extensions import Never

        def capybara(x: Never, y: str, z: int):
            assert x == 1
            assert 1 == 2
            assert __name__ == "__main__"
            assert y == z  # E: unsafe_comparison

    @assert_passes()
    def test_none(self):
        def capybara(x: str):
            assert x is not None

    @assert_passes()
    def test_fancy_none(self):
        class X:
            def __init__(self) -> None:
                self.y = None

        def capybara(x: X):
            assert x.y == 42  # OK

    @assert_passes()
    def test_union(self):
        from typing import Union

        def capybara(x: Union[int, str], z: Union[str, bytes]):
            assert x == z  # ok
            assert x == b"y"  # E: unsafe_comparison
            assert 1 == z  # E: unsafe_comparison

    @assert_passes()
    def test_subclass_value(self):
        def capybara(x: type, y: type[int], marker: int):
            if marker == 0:
                assert x == int
            elif marker == 1:
                assert int == x
            elif marker == 2:
                assert x == y
            elif marker == 3:
                assert y == x
            elif marker == 4:
                assert str == y  # E: unsafe_comparison
            elif marker == 5:
                assert y == str  # E: unsafe_comparison


class TestOverrideEq(TestNameCheckVisitorBase):
    @assert_passes()
    def test_simple_eq(self):
        from typing_extensions import Literal, overload

        class HasSimpleEq:
            def __eq__(self, other: object) -> bool:
                return self is other

        class FancyEq1:
            @overload
            def __eq__(self, x: int) -> Literal[False]: ...
            @overload
            def __eq__(self, x: str) -> Literal[False]: ...
            def __eq__(self, x: object) -> bool:
                return False

        class FancyEq2:
            def __eq__(self, x: object, extra_arg: bool = False) -> bool:
                return False

        class FancyEq3:
            def __eq__(self, *args: object) -> bool:
                return False

        def capybara(
            x: HasSimpleEq, y: int, fe1: FancyEq1, fe2: FancyEq2, fe3: FancyEq3
        ):
            assert x == y  # E: unsafe_comparison
            assert y == x  # E: unsafe_comparison
            assert fe1 == y  # OK
            assert fe2 == y  # OK
            assert fe3 == y  # OK
