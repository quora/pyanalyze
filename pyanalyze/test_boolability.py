# static analysis: ignore
from qcore.asserts import assert_eq


from .error_code import ErrorCode
from .asynq_checker import (
    is_impure_async_fn,
    _stringify_async_fn,
    get_pure_async_equivalent,
)
from .stacked_scopes import Composite
from .tests import (
    PropertyObject,
    async_fn,
    cached_fn,
    proxied_fn,
    l0cached_async_fn,
    Subclass,
    ASYNQ_METHOD_NAME,
)
from .value import KnownValue, UnboundMethodValue, TypedValue
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_fails, assert_passes


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
