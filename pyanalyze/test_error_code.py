# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestRegisterErrorCode(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        from pyanalyze.tests import custom_code

        custom_code()  # E: internal_test
