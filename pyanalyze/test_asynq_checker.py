# static analysis: ignore
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


class TestImpureAsyncCalls(TestNameCheckVisitorBase):
    @assert_passes()
    def test_async_classmethod(self):
        from asynq import asynq, result
        from pyanalyze.tests import PropertyObject, CheckedForAsynq

        class HostQuestion(CheckedForAsynq):
            def init(self, qid):
                self.qid = qid

            @asynq()
            def tree(self):
                yield PropertyObject.async_classmethod.asynq(self.qid)
                result([])

    def test_async_before_classmethod(self):
        # the decorators should work in either order
        from asynq import asynq, result

        class AsyncBeforeClassmethod(object):
            @asynq()
            @classmethod
            def get_context_tid(cls, qid):
                return 4

            @asynq(pure=True)
            @classmethod
            def get_pure_context_tid(cls, qid):
                return 5

        @asynq()
        def async_capybara(qid):
            tid1, tid2 = (
                yield AsyncBeforeClassmethod.get_context_tid.asynq(qid),
                AsyncBeforeClassmethod.get_pure_context_tid(qid),
            )
            result(tid1, tid2)

    @assert_passes()
    def test_async_staticmethod(self):
        from asynq import asynq, result
        from pyanalyze.tests import PropertyObject, CheckedForAsynq

        class CapybaraLink(CheckedForAsynq):
            def init(self, uid):
                self.uid = uid

            @asynq()
            def tree(self):
                log = yield PropertyObject.async_staticmethod.asynq()
                result(str(log))

    @assert_fails(ErrorCode.impure_async_call)
    def test_impure_async_staticmethod(self):
        from pyanalyze.tests import PropertyObject, CheckedForAsynq

        class CapybaraLink(CheckedForAsynq):
            def init(self, uid):
                self.uid = uid

            def tree(self):
                log = PropertyObject.async_staticmethod()
                return str(log)

    @assert_fails(ErrorCode.impure_async_call)
    def test_impure_async_property_access(self):
        from pyanalyze.tests import PropertyObject
        from asynq import asynq

        @asynq()
        def get_capybara(qid):
            po = PropertyObject(qid)
            return po.prop_with_get

    @assert_passes()
    def test_async_property_access(self):
        from pyanalyze.tests import PropertyObject
        from asynq import asynq, result

        @asynq()
        def get_capybara(qid):
            po = PropertyObject(qid)
            result((yield po.get_prop_with_get.asynq()))

    @assert_passes()
    def test_async_attribute_access(self):
        from asynq import asynq

        class Capybara(object):
            def __init__(self, grass):
                self.grass = grass

            @asynq()
            def get_grass(self):
                return self.grass

        @asynq()
        def feed_it(grass):
            capybara = Capybara(grass)
            return capybara.grass

    def test_get_async(self):
        self.assert_is_changed(
            """
from pyanalyze.tests import ClassWithAsync
from asynq import asynq

@asynq()
def capybara():
    return ClassWithAsync().get()
""",
            """
from pyanalyze.tests import ClassWithAsync
from asynq import asynq

@asynq()
def capybara():
    return (yield ClassWithAsync().get_async())
""",
        )

    @assert_passes()
    def test_pure_async_call(self):
        from asynq import asynq, result
        from pyanalyze.tests import async_fn, CheckedForAsynq

        class Capybara(CheckedForAsynq):
            def init(self, qid):
                self.qid = qid

            @asynq()
            def tree(self):
                yield async_fn.asynq(self.qid)
                result([])

    @assert_fails(ErrorCode.impure_async_call)
    def test_impure_async_call(self):
        from pyanalyze.tests import async_fn, CheckedForAsynq

        class Capybara(CheckedForAsynq):
            def init(self, aid):
                self.aid = aid

            def tree(self):
                async_fn(self.aid)
                return []

    @assert_fails(ErrorCode.impure_async_call)
    def test_impure_cached_call(self):
        from pyanalyze.tests import cached_fn, CheckedForAsynq

        class Capybara(CheckedForAsynq):
            def init(self, uid):
                self.uid = uid

            def tree(self):
                cached_fn(self.uid)
                return []

    @assert_fails(ErrorCode.impure_async_call)
    def test_impure_async_call_in_component(self):
        from pyanalyze.tests import cached_fn, CheckedForAsynq
        from asynq import asynq

        class Capybara(CheckedForAsynq):
            def init(self, uid):
                self.uid = uid

            @asynq()
            def tree(self):
                cached_fn(self.uid)
                return []

    @assert_fails(ErrorCode.impure_async_call)
    def test_impure_async_call_to_method(self):
        from asynq import asynq
        from pyanalyze.tests import CheckedForAsynq

        class Capybara(CheckedForAsynq):
            @asynq()
            def render_stuff(self):
                return []

            @asynq()
            def tree(self):
                z = []
                z += self.render_stuff()
                return z

    @assert_fails(ErrorCode.impure_async_call)
    def test_impure_async_for_attributes(self):
        from pyanalyze.tests import PropertyObject, CheckedForAsynq

        class Capybara(CheckedForAsynq):
            def init(self, qid):
                self.qid = qid

            def tree(self):
                PropertyObject(self.qid).prop_with_get
                return []

    @assert_passes()
    def test_pure_async_for_attributes(self):
        from asynq import asynq, result
        from pyanalyze.tests import PropertyObject, CheckedForAsynq

        class Capybara(CheckedForAsynq):
            def init(self, qid):
                self.qid = qid

            @asynq()
            def tree(self):
                yield PropertyObject(self.qid).get_prop_with_get.asynq()
                result([])

    @assert_passes()
    def test_untyped_attribute_accesses(self):
        from pyanalyze.tests import PropertyObject, CheckedForAsynq

        class Capybara(CheckedForAsynq):
            def init(self, qid):
                self.qid = qid

            def get_url(self, question):
                return question.prop

            def tree(self):
                question = PropertyObject(self.qid)
                return self.get_url(question)

    @assert_passes()
    def test_access_in_classmethod(self):
        from pyanalyze.tests import PropertyObject, CheckedForAsynq

        class Capybara(CheckedForAsynq):
            def init(self, qid):
                self.qid = qid

            @classmethod
            def get_url(cls, qid):
                return PropertyObject(qid).prop

            def tree(self):
                return self.get_url(self.qid)

    @assert_passes()
    def test_access_in_property(self):
        from pyanalyze.tests import PropertyObject, CheckedForAsynq

        class LinkImageNonLive(CheckedForAsynq):
            embed_object = property(lambda self: PropertyObject(self.lid).prop)

            def init(self, poid):
                self.poid = poid

    @assert_passes()
    def test_no_error_in_classmethod(self):
        from pyanalyze.tests import cached_fn, CheckedForAsynq

        class Capybara(CheckedForAsynq):
            @classmethod
            def add(cls, tid):
                print(cached_fn(tid))

    @assert_passes()
    def test_lazy_constant(self):
        import qcore

        @qcore.caching.lazy_constant
        def get_capybara():
            return "Neochoerus"

        def fn():
            return get_capybara()

    @assert_fails(ErrorCode.impure_async_call)
    def test_function(self):
        from asynq import asynq
        from pyanalyze.tests import async_fn

        @asynq()
        def capybara(aid):
            return async_fn(aid)

    @assert_fails(ErrorCode.impure_async_call)
    def test_method(self):
        from asynq import asynq
        from pyanalyze.tests import PropertyObject

        @asynq()
        def capybara(aid):
            po = PropertyObject(aid)
            return po.async_method()

    @assert_fails(ErrorCode.impure_async_call)
    def test_classmethod(self):
        from asynq import asynq
        from pyanalyze.tests import PropertyObject

        @asynq()
        def capybara(aid):
            return PropertyObject.async_classmethod(aid)


class TestAsyncMethods(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        from asynq import asynq, result
        from pyanalyze.tests import PropertyObject, CheckedForAsynq

        class Capybara(CheckedForAsynq):
            def init(self, poid):
                self.poid = poid

            @asynq()
            def tree(self):
                z = []
                z += yield PropertyObject(self.poid).async_method.asynq()
                result(z)

    @assert_fails(ErrorCode.impure_async_call)
    def test_impure_method_call(self):
        from pyanalyze.tests import PropertyObject, CheckedForAsynq

        class Capybara(CheckedForAsynq):
            def init(self, poid):
                self.poid = poid

            def tree(self):
                z = []
                z += PropertyObject(self.poid).async_method()
                return z

    @assert_passes()
    def test_pure_asynq_method(self):
        from asynq import asynq
        import multiprocessing

        class Capybara(object):
            def __init__(self, x):
                self.x = x

            def get(self):
                return int(self.x)

            @asynq(pure=True)
            def get_async(self):
                return int(self.x)

        @asynq()
        def use_capybara(x):
            cap = Capybara(x)
            assert_is_value((yield cap.get_async()), TypedValue(int))
            pool = multiprocessing.Pool()
            # should not try to use map_async
            pool.map(len, [(1,)])


def test_stringify_async_fn():
    def check(expected, val):
        assert "pyanalyze.tests." + expected == _stringify_async_fn(val)

    check("async_fn", KnownValue(async_fn))
    check("async_fn." + ASYNQ_METHOD_NAME, KnownValue(async_fn.asynq))
    check(
        "PropertyObject.async_classmethod", KnownValue(PropertyObject.async_classmethod)
    )
    check(
        "PropertyObject.async_classmethod." + ASYNQ_METHOD_NAME,
        KnownValue(PropertyObject.async_classmethod.asynq),
    )
    check("cached_fn", KnownValue(cached_fn))
    check("cached_fn." + ASYNQ_METHOD_NAME, KnownValue(cached_fn.asynq))
    check("proxied_fn", KnownValue(proxied_fn))
    check("l0cached_async_fn", KnownValue(l0cached_async_fn))
    check("l0cached_async_fn." + ASYNQ_METHOD_NAME, KnownValue(l0cached_async_fn.asynq))
    check("PropertyObject.async_method", KnownValue(PropertyObject(1).async_method))
    check(
        "PropertyObject.async_method." + ASYNQ_METHOD_NAME,
        KnownValue(PropertyObject(1).async_method.asynq),
    )

    # UnboundMethodValue
    check(
        "PropertyObject.async_method",
        UnboundMethodValue("async_method", Composite(TypedValue(PropertyObject))),
    )
    check(
        "PropertyObject.async_method.asynq",
        UnboundMethodValue(
            "async_method", Composite(TypedValue(PropertyObject)), "asynq"
        ),
    )

    check(
        "Subclass.async_method", KnownValue(super(Subclass, Subclass(1)).async_method)
    )
    assert "super(pyanalyze.tests.Subclass, self).async_method" == _stringify_async_fn(
        UnboundMethodValue(
            "async_method", Composite(TypedValue(super(Subclass, Subclass)))
        )
    )


def test_is_impure_async_fn():
    assert is_impure_async_fn(KnownValue(async_fn))
    assert not is_impure_async_fn(KnownValue(async_fn.asynq))
    assert is_impure_async_fn(KnownValue(PropertyObject.async_classmethod))
    assert not is_impure_async_fn(KnownValue(PropertyObject.async_classmethod.asynq))
    assert is_impure_async_fn(KnownValue(cached_fn))
    assert not is_impure_async_fn(KnownValue(cached_fn.asynq))
    assert is_impure_async_fn(KnownValue(proxied_fn))
    assert not is_impure_async_fn(KnownValue(proxied_fn.asynq))
    assert is_impure_async_fn(KnownValue(l0cached_async_fn))
    assert not is_impure_async_fn(KnownValue(l0cached_async_fn.asynq))
    assert is_impure_async_fn(KnownValue(PropertyObject(1).async_method))
    assert not is_impure_async_fn(KnownValue(PropertyObject(1).async_method.asynq))

    # UnboundMethodValue
    assert is_impure_async_fn(
        UnboundMethodValue("async_method", Composite(TypedValue(PropertyObject)))
    )
    assert not is_impure_async_fn(
        UnboundMethodValue(
            "async_method", Composite(TypedValue(PropertyObject)), "asynq"
        )
    )


def test_get_pure_async_equivalent():
    known_values = [
        async_fn,
        PropertyObject.async_classmethod,
        cached_fn,
        proxied_fn,
        l0cached_async_fn,
        PropertyObject(1).async_method,
    ]
    for fn in known_values:
        expected = "{}.asynq".format(_stringify_async_fn(KnownValue(fn)))
        assert expected == get_pure_async_equivalent(KnownValue(fn))

    assert (
        "pyanalyze.tests.PropertyObject.async_method.asynq"
        == get_pure_async_equivalent(
            UnboundMethodValue("async_method", Composite(TypedValue(PropertyObject)))
        )
    )
