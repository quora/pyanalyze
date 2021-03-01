# static analysis: ignore
from .arg_spec import assert_is_value
from .value import KnownValue, MultiValuedValue, TypedValue, UNRESOLVED_VALUE
from .test_node_visitor import assert_passes
from .test_name_check_visitor import TestNameCheckVisitorBase


class TestAttributes(TestNameCheckVisitorBase):
    @assert_passes()
    def test_attrs(self):
        import attr

        @attr.s(frozen=True)
        class Capybara(object):
            value = attr.ib()
            int_value = attr.ib(type=int)

        def kerodon():
            c = Capybara(42, 43)
            assert_is_value(c.value, UNRESOLVED_VALUE)
            assert_is_value(c.int_value, TypedValue(int))

    @assert_passes()
    def test_attribute_in_annotations(self):
        class Capybara:
            capybara_id: int
            kerodon_id: object = None

        def capybara():
            assert_is_value(Capybara.kerodon_id, TypedValue(object))
            c = Capybara()
            return c.capybara_id

    @assert_passes()
    def test_attribute_in_base_class(self):
        from typing import Optional

        union = MultiValuedValue([KnownValue(None), TypedValue(int)])

        class Capybara:
            capybara_id: Optional[int] = None

            @classmethod
            def clsmthd(cls):
                assert_is_value(cls.capybara_id, union)

        class DefiniteCapybara(Capybara):
            capybara_id = 3

            @classmethod
            def clsmthd(cls):
                assert_is_value(cls.capybara_id, KnownValue(3))

        def capybara():
            assert_is_value(Capybara().capybara_id, union)
            assert_is_value(Capybara.capybara_id, union)
            assert_is_value(DefiniteCapybara().capybara_id, KnownValue(3))
            assert_is_value(DefiniteCapybara.capybara_id, KnownValue(3))

    @assert_passes()
    def test_attribute_union(self):
        class A:
            x: int

        class B:
            x: str

        class C(B):
            y: float

        def capybara() -> None:
            assert_is_value(A().x, TypedValue(int))
            assert_is_value(C().y, TypedValue(float))
            assert_is_value(C().x, TypedValue(str))

    @assert_passes()
    def test_name_py3(self):
        def capybara():
            assert_is_value(KnownValue.__name__, KnownValue("KnownValue"))

    @assert_passes()
    def test_attribute_type_inference(self):
        from pyanalyze.tests import PropertyObject

        class Capybara(object):
            def init(self, aid):
                self.answer = PropertyObject(aid)

            def tree(self):
                assert_is_value(self.answer, TypedValue(PropertyObject))
                return []

    @assert_passes()
    def test_property_on_unhashable_object(self):
        class CustomDescriptor(object):
            __hash__ = None

            def __get__(self, obj, typ):
                if obj is None:
                    return self
                return 3

        class Unhashable(object):
            __hash__ = None

            prop = CustomDescriptor()

        def use_it():
            assert_is_value(Unhashable().prop, UNRESOLVED_VALUE)

    @assert_passes()
    def test_tuple_subclass_with_getattr(self):

        # Inspired by pyspark.sql.types.Row
        class Row(tuple):
            def __getattr__(self, attr):
                return attr.upper()

        def capybara():
            x = Row()
            return x.capybaras
