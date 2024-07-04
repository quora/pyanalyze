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
    def test_union(self):
        from typing import Union

        def capybara(x: Union[int, str], z: Union[str, bytes]):
            assert x == z  # ok
            assert x == b"y"  # E: unsafe_comparison
            assert 1 == z  # E: unsafe_comparison
