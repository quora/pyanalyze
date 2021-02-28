# static analysis: ignore
from .error_code import ErrorCode
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_fails, assert_passes


class TestMethodReturnType(TestNameCheckVisitorBase):
    @assert_fails(ErrorCode.invalid_method_return_type)
    def test_should_return_nothing(self):
        from pyanalyze.tests import FixedMethodReturnType

        class Capybara(FixedMethodReturnType):
            def should_return_none(self):
                return "this should not return anything but it does"

    @assert_fails(ErrorCode.invalid_method_return_type)
    def test_wrong_type(self):
        from pyanalyze.tests import FixedMethodReturnType

        class Capybara(FixedMethodReturnType):
            def should_return_list(self):
                return "this should return a list, not a string"

    @assert_fails(ErrorCode.invalid_method_return_type)
    def test_multi_valued(self):
        from pyanalyze.tests import FixedMethodReturnType

        def some_condition():
            pass

        class Capybara(FixedMethodReturnType):
            def should_return_list(self):
                if some_condition():
                    return "this should return a list, not a string"
                else:
                    return []

    @assert_fails(ErrorCode.invalid_method_return_type)
    def test_should_return_something_but_does_nothing(self):
        from pyanalyze.tests import FixedMethodReturnType

        class Capybara(FixedMethodReturnType):
            def should_return_list(self):
                pass

    @assert_fails(ErrorCode.invalid_method_return_type)
    def test_should_return_list(self):
        from pyanalyze.tests import FixedMethodReturnType

        class Capybara(FixedMethodReturnType):
            def should_return_list(self):
                return

    @assert_passes()
    def test_no_crash_on_bad_class(self):
        from pyanalyze.tests import FixedMethodReturnType

        class HasAnyAttr(object):
            def __getattr__(self, attr):
                return object()

        def bad_decorator(cls):
            return HasAnyAttr()

        @bad_decorator
        class Capybara(FixedMethodReturnType):
            def should_return_list(self):
                return
