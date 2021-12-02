# static analysis: ignore

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import (
    AnySource,
    AnyValue,
    CallableValue,
    GenericValue,
    KnownValue,
    TypedValue,
    assert_is_value,
)


class TestSyntheticType(TestNameCheckVisitorBase):
    @assert_passes()
    def test_functools(self):
        import functools
        import types
        from pyanalyze.signature import Signature

        sig = Signature.make(
            [], is_ellipsis_args=True, return_annotation=TypedValue(int)
        )

        def f() -> int:
            return 0

        def capybara():
            c = functools.singledispatch(f)
            assert_is_value(
                c, GenericValue("functools._SingleDispatchCallable", [TypedValue(int)])
            )
            assert_is_value(
                c.registry,
                GenericValue(
                    types.MappingProxyType,
                    [AnyValue(AnySource.explicit), CallableValue(sig)],
                ),
            )
            assert_is_value(c._clear_cache(), KnownValue(None))
            assert_is_value(c(), TypedValue(int))
            c.doesnt_exist  # E: undefined_attribute

    @assert_passes()
    def test_protocol(self):
        import csv
        import io

        def capybara():
            writer = io.StringIO()
            assert_is_value(csv.writer(writer), TypedValue("_csv._writer"))
