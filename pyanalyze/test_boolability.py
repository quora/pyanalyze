# static analysis: ignore
from asynq.futures import FutureBase
from qcore.asserts import assert_eq

from .boolability import Boolability, get_boolability
from .stacked_scopes import Composite
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    DictIncompleteValue,
    KnownValue,
    SequenceIncompleteValue,
    TypedDictValue,
    UnboundMethodValue,
    TypedValue,
)
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class BadBool:
    def __bool__(self):
        raise Exception("fooled ya")


class HasLen:
    def __len__(self) -> int:
        return 42


def test_get_boolability() -> None:
    future = TypedValue(FutureBase)
    assert_eq(Boolability.boolable, get_boolability(AnyValue(AnySource.unannotated)))
    assert_eq(
        Boolability.type_always_true,
        get_boolability(UnboundMethodValue("method", Composite(TypedValue(int)))),
    )
    assert_eq(
        Boolability.boolable,
        get_boolability(
            UnboundMethodValue(
                "method", Composite(TypedValue(int)), secondary_attr_name="whatever"
            )
        ),
    )

    # Sequence/dict values
    assert_eq(
        Boolability.type_always_true,
        get_boolability(TypedDictValue({"a": (True, TypedValue(int))})),
    )
    assert_eq(
        Boolability.boolable,
        get_boolability(TypedDictValue({"a": (False, TypedValue(int))})),
    )
    assert_eq(
        Boolability.type_always_true,
        get_boolability(SequenceIncompleteValue(tuple, [KnownValue(1)])),
    )
    assert_eq(
        Boolability.value_always_false,
        get_boolability(SequenceIncompleteValue(tuple, [])),
    )
    assert_eq(
        Boolability.value_always_true_mutable,
        get_boolability(SequenceIncompleteValue(list, [KnownValue(1)])),
    )
    assert_eq(
        Boolability.value_always_false_mutable,
        get_boolability(SequenceIncompleteValue(list, [])),
    )
    assert_eq(
        Boolability.value_always_true_mutable,
        get_boolability(DictIncompleteValue(dict, [(KnownValue(1), KnownValue(1))])),
    )
    assert_eq(
        Boolability.value_always_false_mutable,
        get_boolability(DictIncompleteValue(dict, [])),
    )

    # KnownValue
    assert_eq(Boolability.erroring_bool, get_boolability(KnownValue(BadBool())))
    assert_eq(Boolability.value_always_true, get_boolability(KnownValue(1)))
    assert_eq(Boolability.type_always_true, get_boolability(KnownValue(int)))
    assert_eq(Boolability.value_always_false, get_boolability(KnownValue(0)))

    # TypedValue
    assert_eq(Boolability.boolable, get_boolability(TypedValue(HasLen)))
    assert_eq(Boolability.erroring_bool, get_boolability(future))
    assert_eq(Boolability.type_always_true, get_boolability(TypedValue(object)))
    assert_eq(Boolability.boolable, get_boolability(TypedValue(int)))

    # MultiValuedValue and AnnotatedValue
    assert_eq(
        Boolability.erroring_bool,
        get_boolability(AnnotatedValue(future, [KnownValue(1)])),
    )
    assert_eq(Boolability.erroring_bool, get_boolability(future | KnownValue(1)))
    assert_eq(Boolability.boolable, get_boolability(TypedValue(int) | TypedValue(str)))
    assert_eq(Boolability.boolable, get_boolability(TypedValue(int) | KnownValue("")))
    assert_eq(
        Boolability.boolable, get_boolability(KnownValue(True) | KnownValue(False))
    )
    assert_eq(
        Boolability.boolable, get_boolability(TypedValue(type) | KnownValue(False))
    )
    assert_eq(
        Boolability.value_always_true,
        get_boolability(TypedValue(type) | KnownValue(True)),
    )
    assert_eq(
        Boolability.value_always_true_mutable,
        get_boolability(TypedValue(type) | KnownValue([1])),
    )
    assert_eq(
        Boolability.value_always_true_mutable,
        get_boolability(KnownValue([1]) | KnownValue(True)),
    )
    assert_eq(
        Boolability.value_always_false_mutable,
        get_boolability(KnownValue(False) | KnownValue([])),
    )
    assert_eq(
        Boolability.value_always_false,
        get_boolability(KnownValue(False) | KnownValue("")),
    )


class TestAssert(TestNameCheckVisitorBase):
    @assert_passes()
    def test_assert_never_fails(self):
        def capybara():
            tpl = "this", "doesn't", "work"
            assert tpl  # E: type_always_true

    @assert_passes()
    def test_assert_bad_bool(self):
        class X(object):
            def __bool__(self):
                raise Exception("I am a poorly behaved object")

            __nonzero__ = __bool__

        x = X()

        def capybara():
            assert x  # E: type_does_not_support_bool


class TestConditionAlwaysTrue(TestNameCheckVisitorBase):
    @assert_passes()
    def test_method(self):
        class Capybara(object):
            def eat(self):
                pass

            def maybe_eat(self):
                if self.eat:  # E: type_always_true
                    self.eat()

    @assert_passes()
    def test_typed_value(self):
        class Capybara(object):
            pass

        if Capybara():  # E: type_always_true
            pass

    @assert_passes()
    def test_overrides_len(self):
        class Capybara(object):
            def __len__(self):
                return 42

        if Capybara():
            pass

    @assert_passes()
    def test_object():
        def capybara():
            True if object() else False  # E: type_always_true
            object() and False  # E: type_always_true
            [] and object() and False  # E: type_always_true
            object() or True  # E: type_always_true
            not object()  # E: type_always_true

    @assert_passes()
    def test_async_yield_or(self):
        from asynq import asynq

        @asynq()
        def kerodon():
            return 42

        @asynq()
        def capybara():
            yield kerodon.asynq() or None  # E: type_does_not_support_bool
