# static analysis: ignore
from .suggested_type import prepare_type
from .value import KnownValue, SubclassValue, TypedValue
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


class A:
    pass


class B(A):
    pass


class C(A):
    pass


def test_prepare_type() -> None:
    assert prepare_type(KnownValue(int) | KnownValue(str)) == TypedValue(type)
    assert prepare_type(KnownValue(C) | KnownValue(B)) == SubclassValue(TypedValue(A))
    assert prepare_type(KnownValue(int)) == SubclassValue(TypedValue(int))

    assert prepare_type(SubclassValue(TypedValue(B)) | KnownValue(C)) == SubclassValue(
        TypedValue(A)
    )
    assert prepare_type(SubclassValue(TypedValue(B)) | KnownValue(B)) == SubclassValue(
        TypedValue(B)
    )
    assert prepare_type(KnownValue(None) | TypedValue(str)) == KnownValue(
        None
    ) | TypedValue(str)
    assert prepare_type(KnownValue(True) | KnownValue(False)) == TypedValue(bool)
