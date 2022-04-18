# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import GenericValue, KnownValue, TypedValue


class TestPEP673(TestNameCheckVisitorBase):
    @assert_passes()
    def test_instance_attribute(self):
        from typing_extensions import Self

        class X:
            parent: Self

            @property
            def prop(self) -> Self:
                raise NotImplementedError

        class Y(X):
            pass

        def capybara(x: X, y: Y):
            assert_is_value(x.parent, TypedValue(X))
            assert_is_value(y.parent, TypedValue(Y))

            assert_is_value(x.prop, TypedValue(X))
            assert_is_value(y.prop, TypedValue(Y))

    @assert_passes()
    def test_method(self):
        from typing_extensions import Self

        class X:
            def ret(self) -> Self:
                return self

            @classmethod
            def from_config(cls) -> Self:
                return cls()

        class Y(X):
            pass

        def capybara(x: X, y: Y):
            assert_is_value(x.ret(), TypedValue(X))
            assert_is_value(y.ret(), TypedValue(Y))

            assert_is_value(X.from_config(), TypedValue(X))
            assert_is_value(Y.from_config(), TypedValue(Y))

    @assert_passes()
    def test_parameter_type(self):
        from typing import Callable

        from typing_extensions import Self

        class Shape:
            def difference(self, other: Self) -> float:
                raise NotImplementedError

            def apply(self, f: Callable[[Self], None]) -> None:
                raise NotImplementedError

        class Circle(Shape):
            pass

        def difference():
            s = Shape()
            s.difference(s)
            s.difference(1.0)  # E: incompatible_argument
            s.difference(Circle())

            c = Circle()
            c.difference(c)
            c.difference(s)  # E: incompatible_argument
            c.difference("x")  # E: incompatible_argument

        def takes_shape(s: Shape) -> None:
            pass

        def takes_circle(c: Circle) -> None:
            pass

        def takes_int(i: int) -> None:
            pass

        def apply():
            s = Shape()
            c = Circle()
            s.apply(takes_shape)
            s.apply(takes_circle)  # E: incompatible_argument
            s.apply(takes_int)  # E: incompatible_argument
            c.apply(takes_shape)
            c.apply(takes_circle)
            c.apply(takes_int)  # E: incompatible_argument

    @assert_passes()
    def test_linked_list(self):
        from dataclasses import dataclass
        from typing import Generic, Optional, TypeVar

        from typing_extensions import Self

        T = TypeVar("T")

        @dataclass
        class LinkedList(Generic[T]):
            value: T
            next: Optional[Self] = None

        @dataclass
        class OrdinalLinkedList(LinkedList[int]):
            pass

        def capybara(o: OrdinalLinkedList):
            # Unfortunately we don't fully support the example in
            assert_is_value(o.next, KnownValue(None) | TypedValue(OrdinalLinkedList))

    @assert_passes()
    def test_generic(self):
        from typing import Generic, TypeVar

        from typing_extensions import Self

        T = TypeVar("T")

        class Container(Generic[T]):
            value: T

            def set_value(self, value: T) -> Self:
                return self

        def capybara(c: Container[int]):
            assert_is_value(c.value, TypedValue(int))
            assert_is_value(c.set_value(3), GenericValue(Container, [TypedValue(int)]))

    @assert_passes()
    def test_classvar(self):
        from typing import ClassVar, List

        from typing_extensions import Self

        class Registry:
            children: ClassVar[List[Self]]

        def capybara():
            assert_is_value(
                Registry.children, GenericValue(list, [TypedValue(Registry)])
            )

    @assert_passes()
    def test_stub(self):
        def capybara():
            from _pyanalyze_tests.self import X, Y

            x = X()
            y = Y()

            def want_x(x: X):
                pass

            def want_y(y: Y):
                pass

            want_x(x.ret())
            want_y(y.ret())

            want_x(X.from_config())
            want_y(Y.from_config())
