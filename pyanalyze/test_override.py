# static analysis: ignore
from .error_code import ErrorCode
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_fails, assert_passes


class TestOverride(TestNameCheckVisitorBase):
    @assert_fails(ErrorCode.invalid_override_decorator)
    def test_invalid_usage(self):
        from typing_extensions import override

        @override
        def not_a_method():
            pass

    @assert_passes()
    def test_valid_method(self):
        from typing_extensions import override

        class Base:
            def method(self):
                pass

        class Capybara(Base):
            @override
            def method(self):
                pass

    @assert_fails(ErrorCode.override_does_not_override)
    def test_invalid_method(self):
        from typing_extensions import override

        class Base:
            def method(self):
                pass

        class Capybara(Base):
            @override
            def no_base_method(self):  # E: override_does_not_override
                pass
