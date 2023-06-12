# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import KnownValue


class TestImport(TestNameCheckVisitorBase):
    @assert_passes()
    def test_import(self):
        import pyanalyze as P

        def capybara() -> None:
            import pyanalyze
            import pyanalyze as py
            import pyanalyze.extensions as E

            assert_is_value(pyanalyze, KnownValue(P))
            assert_is_value(py, KnownValue(P))
            assert_is_value(E, KnownValue(P.extensions))

    @assert_passes()
    def test_import_from(self):
        def capybara():
            import pyanalyze as P

            def capybara():
                from pyanalyze import extensions
                from pyanalyze.extensions import assert_error

                assert_is_value(extensions, KnownValue(P.extensions))
                assert_is_value(assert_error, KnownValue(P.extensions.assert_error))

    def test_import_star(self):
        self.assert_passes("""
            import pyanalyze as P

            if False:
                from pyanalyze import *

                assert_is_value(extensions, KnownValue(P.extensions))
                not_a_name  # E: undefined_name
            """)


class TestDisallowedImport(TestNameCheckVisitorBase):
    @assert_passes()
    def test_top_level(self):
        import getopt  # E: disallowed_import

        from getopt import getopt  # E: disallowed_import

        def capybara():
            import getopt  # E: disallowed_import
            from getopt import getopt  # E: disallowed_import

    @assert_passes()
    def test_nested(self):
        import pyanalyze._disallowed_test  # E: disallowed_import
        from pyanalyze._disallowed_test import some_func  # E: disallowed_import
        from pyanalyze import _disallowed_test  # E: disallowed_import

        def capybara():
            import pyanalyze._disallowed_test  # E: disallowed_import
            from pyanalyze._disallowed_test import some_func  # E: disallowed_import
            from pyanalyze import _disallowed_test  # E: disallowed_import
