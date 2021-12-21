# static analysis: ignore
from .error_code import ErrorCode
from .test_node_visitor import assert_passes
from .test_name_check_visitor import TestNameCheckVisitorBase


class TestSuggestedType(TestNameCheckVisitorBase):
    @assert_passes(settings={ErrorCode.suggested_return_type: True})
    def test_return(self):
        def capybara():  # E: suggested_return_type
            return 1

        def kerodon(cond):  # E: suggested_return_type
            if cond:
                return 1
            else:
                return 2

    @assert_passes(settings={ErrorCode.suggested_parameter_type: True})
    def test_parameter(self):
        def capybara(a):  # E: suggested_parameter_type
            pass

        def annotated(b: int):
            pass

        class Mammalia:
            # should not suggest a type for this
            def method(self):
                pass

        def kerodon(unannotated):
            capybara(1)
            annotated(2)

            m = Mammalia()
            m.method()
            Mammalia.method(unannotated)
