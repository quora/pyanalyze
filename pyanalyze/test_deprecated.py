# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before


class TestStub(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def capybara():
            print("keep")
            from _pyanalyze_tests.deprecated import DeprecatedCapybara  # E: deprecated

            print("these imports")
            from _pyanalyze_tests.deprecated import deprecated_function  # E: deprecated

            print("separate")
            from _pyanalyze_tests.deprecated import deprecated_overload

            deprecated_overload(1)  # E: deprecated
            deprecated_overload("x")

            deprecated_function(1)
            print(deprecated_function)
            DeprecatedCapybara()
            print(DeprecatedCapybara)

    @skip_before((3, 10))
    @assert_passes()
    def test_multiline_import(self):
        def capybara():
            from _pyanalyze_tests.deprecated import (
                DeprecatedCapybara,  # E: deprecated
                deprecated_function,  # E: deprecated
                deprecated_overload,
            )

            return [deprecated_function, deprecated_overload, DeprecatedCapybara]


class TestRuntime(TestNameCheckVisitorBase):
    @assert_passes()
    def test_overload(self):
        from pyanalyze.extensions import deprecated, overload

        @overload
        @deprecated("int support is deprecated")
        def deprecated_overload(x: int) -> int: ...

        @overload
        def deprecated_overload(x: str) -> str: ...

        def deprecated_overload(x):
            return x

        def capybara():
            deprecated_overload(1)  # E: deprecated
            deprecated_overload("x")

    @assert_passes()
    def test_function(self):
        from pyanalyze.extensions import deprecated

        @deprecated("no functioning capybaras")
        def deprecated_function(x: int) -> int:
            return x

        def capybara():
            print(deprecated_function)  # E: deprecated
            deprecated_function(1)  # E: deprecated

    @assert_passes()
    def test_method(self):
        from pyanalyze.extensions import deprecated

        class Cls:
            @deprecated("no methodical capybaras")
            def deprecated_method(self, x: int) -> int:
                return x

        def capybara():
            Cls().deprecated_method(1)  # E: deprecated
            print(Cls.deprecated_method)  # E: deprecated

    @assert_passes()
    def test_class(self):
        from pyanalyze.extensions import deprecated

        @deprecated("no classy capybaras")
        class DeprecatedClass:
            pass

        def capybara():
            print(DeprecatedClass)  # E: deprecated
            return DeprecatedClass()  # E: deprecated
