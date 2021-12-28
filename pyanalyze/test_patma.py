# static analysis: ignore
from .test_node_visitor import skip_before
from .test_name_check_visitor import TestNameCheckVisitorBase


class TestPatma(TestNameCheckVisitorBase):
    @skip_before((3, 10))
    def test_singletons(self):
        self.assert_passes(
            """
            from typing import Literal
            def capybara(x: Literal[True, False, None]):
                match x:
                    case True:
                        assert_is_value(x, KnownValue(True))
                    case _:
                        assert_is_value(x, KnownValue(False) | KnownValue(None))
            """
        )
