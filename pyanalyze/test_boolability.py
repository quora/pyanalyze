# static analysis: ignore
from asynq.futures import FutureBase

from .boolability import Boolability, get_boolability
from .stacked_scopes import Composite
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    DictIncompleteValue,
    KnownValue,
    KVPair,
    NO_RETURN_VALUE,
    SequenceValue,
    TypedDictValue,
    TypedValue,
    UnboundMethodValue,
)


class BadBool:
    def __bool__(self):
        raise Exception("fooled ya")


class HasLen:
    def __len__(self) -> int:
        return 42


def test_get_boolability() -> None:
    future = TypedValue(FutureBase)
    assert Boolability.boolable == get_boolability(AnyValue(AnySource.unannotated))
    assert Boolability.type_always_true == get_boolability(
        UnboundMethodValue("method", Composite(TypedValue(int)))
    )
    assert Boolability.boolable == get_boolability(
        UnboundMethodValue(
            "method", Composite(TypedValue(int)), secondary_attr_name="whatever"
        )
    )

    # Sequence/dict values
    assert Boolability.type_always_true == get_boolability(
        TypedDictValue({"a": (True, TypedValue(int))})
    )
    assert Boolability.boolable == get_boolability(
        TypedDictValue({"a": (False, TypedValue(int))})
    )
    assert Boolability.type_always_true == get_boolability(
        SequenceValue(tuple, [(False, KnownValue(1))])
    )
    assert Boolability.value_always_false == get_boolability(SequenceValue(tuple, []))
    assert Boolability.boolable == get_boolability(
        SequenceValue(tuple, [(True, KnownValue(1))])
    )
    assert Boolability.type_always_true == get_boolability(
        # many 1s followed by one 2
        SequenceValue(tuple, [(True, KnownValue(1)), (False, KnownValue(2))])
    )
    assert Boolability.value_always_true_mutable == get_boolability(
        SequenceValue(list, [(False, KnownValue(1))])
    )
    assert Boolability.value_always_false_mutable == get_boolability(
        SequenceValue(list, [])
    )
    assert Boolability.boolable == get_boolability(
        SequenceValue(list, [(True, KnownValue(1))])
    )
    assert Boolability.value_always_true_mutable == get_boolability(
        # many 1s followed by one 2
        SequenceValue(list, [(True, KnownValue(1)), (False, KnownValue(2))])
    )

    assert Boolability.value_always_true_mutable == get_boolability(
        DictIncompleteValue(dict, [KVPair(KnownValue(1), KnownValue(1))])
    )
    assert Boolability.boolable == get_boolability(
        DictIncompleteValue(
            dict, [KVPair(KnownValue(1), KnownValue(1), is_required=False)]
        )
    )
    assert Boolability.boolable == get_boolability(
        DictIncompleteValue(
            dict, [KVPair(TypedValue(int), KnownValue(1), is_many=True)]
        )
    )
    assert Boolability.value_always_false_mutable == get_boolability(
        DictIncompleteValue(dict, [])
    )

    # KnownValue
    assert Boolability.erroring_bool == get_boolability(KnownValue(BadBool()))
    assert Boolability.value_always_true == get_boolability(KnownValue(1))
    assert Boolability.type_always_true == get_boolability(KnownValue(int))
    assert Boolability.value_always_false == get_boolability(KnownValue(0))

    # TypedValue
    assert Boolability.boolable == get_boolability(TypedValue(HasLen))
    assert Boolability.erroring_bool == get_boolability(future)
    assert Boolability.boolable == get_boolability(TypedValue(object))
    assert Boolability.boolable == get_boolability(TypedValue(int))
    assert Boolability.boolable == get_boolability(TypedValue(object))

    # MultiValuedValue and AnnotatedValue
    assert Boolability.erroring_bool == get_boolability(
        AnnotatedValue(future, [KnownValue(1)])
    )
    assert Boolability.erroring_bool == get_boolability(future | KnownValue(1))
    assert Boolability.boolable == get_boolability(TypedValue(int) | TypedValue(str))
    assert Boolability.boolable == get_boolability(TypedValue(int) | KnownValue(""))
    assert Boolability.boolable == get_boolability(KnownValue(True) | KnownValue(False))
    assert Boolability.boolable == get_boolability(TypedValue(type) | KnownValue(False))
    assert Boolability.value_always_true == get_boolability(
        TypedValue(type) | KnownValue(True)
    )
    assert Boolability.value_always_true_mutable == get_boolability(
        TypedValue(type) | KnownValue([1])
    )
    assert Boolability.value_always_true_mutable == get_boolability(
        KnownValue([1]) | KnownValue(True)
    )
    assert Boolability.value_always_false_mutable == get_boolability(
        KnownValue(False) | KnownValue([])
    )
    assert Boolability.value_always_false == get_boolability(
        KnownValue(False) | KnownValue("")
    )
    assert Boolability.boolable == get_boolability(NO_RETURN_VALUE)


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
        obj = object()

        def capybara():
            True if obj else False  # E: type_always_true
            obj and False  # E: type_always_true
            [] and obj and False  # E: type_always_true
            obj or True  # E: type_always_true
            not obj  # E: type_always_true

    @assert_passes()
    def test_async_yield_or(self):
        from asynq import asynq

        @asynq()
        def kerodon():
            return 42

        @asynq()
        def capybara():
            yield kerodon.asynq() or None  # E: type_does_not_support_bool
