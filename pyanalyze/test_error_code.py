# static analysis: ignore
from .error_code import register_error_code
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestRegisterErrorCode(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        from pyanalyze.tests import custom_code

        custom_code()  # E: internal_test


def test_register_error_code_twice() -> None:
    first = register_error_code("internal_test_twice", "Registered twice in a test")
    second = register_error_code("internal_test_twice", "Registered twice in a test")
    assert first is second
