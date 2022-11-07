# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestOwnership(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import List

        def capybara(x: List[str]):
            x.append("x")  # E: disallowed_mutation
            y = list(x)
            y.append("x")

            z = [a for a in x]
            z.append("x")

            # TODO make it work for an all-literal list
            alpha = ["a", "b", "c", str(x)]
            alpha.append("x")
