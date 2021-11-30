# static analysis: ignore
from typing import Dict, Union
from .error_code import ErrorCode
from .value import (
    AnySource,
    AnyValue,
    GenericValue,
    KnownValue,
    MultiValuedValue,
    TypedValue,
    assert_is_value,
)
from .test_node_visitor import assert_passes, assert_fails
from .test_name_check_visitor import TestNameCheckVisitorBase

_global_dict: Dict[Union[int, str], float] = {}


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
            assert_is_value(c.value, AnyValue(AnySource.unannotated))
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
    def test_generic(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")
        U = TypeVar("U")

        class X(Generic[T]):
            x: T

        class Child1(X[str]):
            pass

        class Child2(X[U]):
            pass

        def capybara(obj: X[int], c1: Child1, c2: Child2[bool]) -> None:
            assert_is_value(obj.x, TypedValue(int))
            assert_is_value(c1.x, TypedValue(str))
            assert_is_value(c2.x, TypedValue(bool))

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
            assert_is_value(Unhashable().prop, AnyValue(AnySource.inference))

    @assert_passes()
    def test_tuple_subclass_with_getattr(self):

        # Inspired by pyspark.sql.types.Row
        class Row(tuple):
            def __getattr__(self, attr):
                return attr.upper()

        def capybara():
            x = Row()
            return x.capybaras

    @assert_passes()
    def test_union(self):
        from typing import Union
        from dataclasses import dataclass

        @dataclass
        class Capybara:
            attr: int

        @dataclass
        class Paca:
            attr: str

        def test(x: Union[Capybara, Paca]) -> None:
            assert_is_value(
                x.attr, MultiValuedValue([TypedValue(int), TypedValue(str)])
            )

    @assert_fails(ErrorCode.unsupported_operation)
    def test_optional_operation(self):
        from typing import Optional

        def capybara(x: Optional[str]):
            print(x[1:])

    @assert_fails(ErrorCode.undefined_attribute)
    def test_optional(self):
        from typing import Optional

        def capybara(x: Optional[str]):
            x.split()

    @assert_passes()
    def test_typeshed(self):
        def capybara(c: staticmethod):
            assert_is_value(c.__isabstractmethod__, TypedValue(bool))

    @assert_fails(ErrorCode.undefined_attribute)
    def test_no_attribute_for_typeshed_class():
        def capybara(c: staticmethod):
            c.no_such_attribute

    @assert_passes()
    def test_typeshed_getattr(self):
        # has __getattribute__ in typeshed
        from types import SimpleNamespace

        # has __getattr__
        from codecs import StreamWriter

        def capybara(sn: SimpleNamespace, sw: StreamWriter):
            assert_is_value(sn.whatever, AnyValue(AnySource.inference))
            assert_is_value(sw.whatever, AnyValue(AnySource.inference))

    @assert_passes()
    def test_allow_function(self):
        def decorator(f):
            return f

        def capybara():
            @decorator
            def f():
                pass

            f.attr = 42
            print(f.attr)

    @assert_passes()
    def test_enum_name(self):
        import enum

        class E(enum.Enum):
            name = 1
            no_name = 2

        def capybara():
            assert_is_value(E.no_name, KnownValue(E.no_name))
            assert_is_value(E.name, KnownValue(E.name))
            E.what_is_this  # E: undefined_attribute

    @assert_passes()
    def test_module_annotations(self):
        from pyanalyze import test_attributes
        from pyanalyze.type_object import TypeObject
        from typing import Optional

        annotated_global: Optional[str] = None

        def capybara():
            assert_is_value(
                test_attributes._global_dict,
                GenericValue(
                    dict, [TypedValue(int) | TypedValue(str), TypedValue(float)]
                ),
            )
            assert_is_value(
                annotated_global, MultiValuedValue([TypedValue(str), KnownValue(None)])
            )


class TestHasAttrExtension(TestNameCheckVisitorBase):
    @assert_passes()
    def test_hasattr(self):
        from typing_extensions import Literal

        def capybara(x: Literal[1]) -> None:
            if hasattr(x, "x"):
                assert_is_value(x.x, AnyValue(AnySource.inference))

    @assert_passes()
    def test_user_hasattr(self):
        from typing import TypeVar, Any
        from typing_extensions import Annotated, Literal
        from pyanalyze.extensions import HasAttrGuard

        T = TypeVar("T", bound=str)

        def my_hasattr(
            obj: object, name: T
        ) -> Annotated[bool, HasAttrGuard["obj", T, Any]]:
            return hasattr(obj, name)

        def has_int_attr(
            obj: object, name: T
        ) -> Annotated[bool, HasAttrGuard["obj", T, int]]:
            val = getattr(obj, name, None)
            return isinstance(val, int)

        def capybara(x: Literal[1]) -> None:
            if my_hasattr(x, "x"):
                assert_is_value(x.x, AnyValue(AnySource.explicit))

        def inty_capybara(x: Literal[1]) -> None:
            if has_int_attr(x, "inty"):
                assert_is_value(x.inty, TypedValue(int))
