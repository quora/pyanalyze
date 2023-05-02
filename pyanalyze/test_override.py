# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestOverride(TestNameCheckVisitorBase):
    @assert_passes()
    def test_invalid_usage(self):
        from typing_extensions import override

        @override
        def not_a_method():
            pass

    @assert_passes()
    def test_method(self):
        from typing_extensions import override

        class Base:
            def method(self):
                pass

        class Capybara(Base):
            @override
            def method(self):
                pass

            @override
            def no_base_method(self):
                pass
