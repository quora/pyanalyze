# static analysis: ignore
import ast

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes

from .yield_checker import _camel_case_to_snake_case, VarnameGenerator


class TestUnnecessaryYield(TestNameCheckVisitorBase):
    @assert_passes()
    def test_failure(self):
        from asynq import asynq, result

        @asynq()
        def inner(n):
            return 1

        @asynq()
        def capybara():
            var1 = yield inner.asynq(1)
            var2 = yield inner.asynq(2)  # E: unnecessary_yield
            result(var1 + var2)

    @assert_passes()
    def test_success(self):
        from asynq import asynq, result

        @asynq()
        def inner(n):
            return 1

        @asynq()
        def capybara():
            var1, var2 = yield (inner.asynq(1), inner.asynq(2))
            result(var1 + var2)

    def test_attribute(self):
        self.assert_is_changed(
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq

            class Capybara(object):
                @asynq()
                def load(self):
                    self.var1 = yield async_fn.asynq(1)
                    self.var2 = yield async_fn.asynq(2)
            """,
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq

            class Capybara(object):
                @asynq()
                def load(self):
                    self.var1, self.var2 = yield async_fn.asynq(1), async_fn.asynq(2)
            """,
        )

    @assert_passes()
    def test_if(self):
        from asynq import asynq, result

        from pyanalyze.tests import PropertyObject

        @asynq()
        def capybara(qid, include_deleted):
            if include_deleted:
                value = yield PropertyObject.load.asynq(qid, include_deleted=False)
            else:
                value = yield PropertyObject.load.asynq(qid, include_deleted=True)
            result(value)

    @assert_passes()
    def test_nested(self):
        from asynq import asynq, result
        from asynq.tools import afilter

        from pyanalyze.tests import async_fn, PropertyObject

        @asynq()
        def capybara(qids, t):
            @asynq()
            def filter_fn(qid):
                result((yield PropertyObject(qid).get_prop_with_get.asynq()) < t)

            qids = yield afilter.asynq(filter_fn, qids)
            system_a2a_ret = yield tuple(async_fn.asynq(qid) for qid in qids)
            result(system_a2a_ret)

    @assert_passes()
    def test_usage_in_nested_function(self):
        from asynq import asynq, result

        from pyanalyze.tests import async_fn, cached_fn

        @asynq()
        def capybara(oid):
            second_oid = yield cached_fn.asynq(oid)
            new_oid = yield async_fn.asynq(second_oid)

            def process():
                return new_oid - 2

            result(process())


class TestUnnecessaryYieldInObject(TestNameCheckVisitorBase):
    @assert_passes()
    def test_across_variable(self):
        from asynq import asynq, result

        class Capybara(object):
            @asynq()
            def render_grass(self):
                return []

            @asynq()
            def render_kerodon(self):
                return []

            @asynq()
            def tree(self):
                grass = yield self.render_grass.asynq()
                z = []
                z += grass
                z += yield self.render_kerodon.asynq()  # E: unnecessary_yield
                result(z)

    @assert_passes()
    def test_basic(self):
        from asynq import asynq, result

        class Capybara(object):
            @asynq()
            def render_grass(self):
                return []

            @asynq()
            def render_kerodon(self):
                return []

            @asynq()
            def tree(self):
                z = []
                z += yield self.render_grass.asynq()
                z += yield self.render_kerodon.asynq()  # E: unnecessary_yield
                result(z)

    @assert_passes()
    def test_in_with(self):
        from asynq import asynq, result

        class Capybara(object):
            @asynq()
            def render_grass(self):
                return []

            @asynq()
            def render_kerodon(self):
                return []

            @asynq()
            def tree(self, ctx):
                z = []
                z += yield self.render_grass.asynq()
                with ctx.into(z):
                    z += yield self.render_kerodon.asynq()
                result(z)


class TestBatchingYields(TestNameCheckVisitorBase):
    """Tests that the replacements produced by the node visitor are correct.

    The unnecessary yield check does the replacements.

    """

    def test_basic(self):
        # also tests that it only fixes one error and stops
        self.assert_is_changed(
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val1 = yield async_fn.asynq(1)
                # some other code
                val2 = 1 + 2 + 3
                # back to yielding
                val3 = yield async_fn.asynq(3)
                val4 = yield async_fn.asynq(4)
                result(val1 + val2 + val3)
            """,
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                # some other code
                val2 = 1 + 2 + 3
                # back to yielding
                val1, val3, val4 = yield async_fn.asynq(1), async_fn.asynq(3), async_fn.asynq(4)
                result(val1 + val2 + val3)
            """,
            repeat=True,
        )

    def test_yield_tuple(self):
        # also tests multiple line assign statement
        self.assert_is_changed(
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val1, val2 = \
                    yield tuple((
                        async_fn.asynq(1),
                        async_fn.asynq(2)
                    ))
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                val4 = yield async_fn.asynq(
                    val3
                )
                result(val1 + val2 + val4)
            """,
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                val1, val2, val4 = yield async_fn.asynq(1), async_fn.asynq(2), async_fn.asynq(val3)
                result(val1 + val2 + val4)
            """,
            repeat=True,
        )

    def test_lots_of_underscores(self):
        self.assert_is_changed(
            """
            from asynq import asynq
            from pyanalyze.tests import async_fn

            @asynq()
            def capybara(oid, oid2):
                _ = yield async_fn.asynq(oid)
                _ = yield async_fn.asynq(oid2)
            """,
            """
            from asynq import asynq
            from pyanalyze.tests import async_fn

            @asynq()
            def capybara(oid, oid2):
                yield async_fn.asynq(oid), async_fn.asynq(oid2)
            """,
        )

    def test_target_tuple(self):
        # when multiple values are yielded to one target
        self.assert_is_changed(
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val1 = yield async_fn.asynq(1), async_fn.asynq(2)
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                val4 = yield async_fn.asynq(4)
                result(val1[1] + val3 + val4)
            """,
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                val1, val4 = yield (async_fn.asynq(1), async_fn.asynq(2)), async_fn.asynq(4)
                result(val1[1] + val3 + val4)
            """,
            repeat=True,
        )
        # same as above but List instead of Tuple
        self.assert_is_changed(
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val1 = yield [async_fn.asynq(1), async_fn.asynq(2)]
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                val4 = yield async_fn.asynq(4)
                result(val1[1] + val3 + val4)
            """,
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                val1, val4 = yield [async_fn.asynq(1), async_fn.asynq(2)], async_fn.asynq(4)
                result(val1[1] + val3 + val4)
            """,
            repeat=True,
        )
        # when target is a Tuple and value is a List
        self.assert_is_changed(
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val1, val2 = yield [async_fn.asynq(1), async_fn.asynq(2)]
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                val4 = yield async_fn.asynq(4)
                result(val1 + val2 + val3 + val4)
            """,
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                (val1, val2), val4 = yield [async_fn.asynq(1), async_fn.asynq(2)], async_fn.asynq(4)
                result(val1 + val2 + val3 + val4)
            """,
            repeat=True,
        )
        # when one value is unwrapped to a target tuple
        self.assert_is_changed(
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val1, val2 = yield async_fn.asynq(1)
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                val4 = yield async_fn.asynq(4)
                result(val1 + val2 + val3 + val4)
            """,
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                (val1, val2), val4 = yield async_fn.asynq(1), async_fn.asynq(4)
                result(val1 + val2 + val3 + val4)
            """,
            repeat=True,
        )

    def test_assign_and_non_assign(self):
        self.assert_is_changed(
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val1 = yield async_fn.asynq(1)
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                val3 += yield async_fn.asynq(4)
                result(val1 + val4)
            """,
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val1, async_fn_result = yield async_fn.asynq(1), async_fn.asynq(4)
                # some other code
                val3 = 1 + 2 + 3
                # back to yielding
                val3 += async_fn_result
                result(val1 + val4)
            """,
            repeat=True,
        )

    def test_non_assign_and_assign(self):
        self.assert_is_changed(
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val3 = 1 + 2 + 3
                val3 += yield async_fn.asynq(1)
                val4 = yield async_fn.asynq(4)
                result(val3 + val4)
            """,
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val3 = 1 + 2 + 3
                async_fn_result, val4 = yield async_fn.asynq(1), async_fn.asynq(4)
                val3 += async_fn_result
                result(val3 + val4)
            """,
            repeat=True,
        )

    def test_both_non_assign(self):
        self.assert_is_changed(
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val3 = 1 + 2 + 3
                val3 += yield async_fn.asynq(1)
                val3 += yield async_fn.asynq(4)
                result(val3)
            """,
            """
            from pyanalyze.tests import async_fn
            from asynq import asynq, result
            @asynq()
            def f():
                val3 = 1 + 2 + 3
                async_fn_result, async_fn_result2 = yield async_fn.asynq(1), async_fn.asynq(4)
                val3 += async_fn_result
                val3 += async_fn_result2
                result(val3)
            """,
            repeat=True,
        )

    @assert_passes()
    def test_in_except_handler(self):
        from asynq import asynq, result

        from pyanalyze.tests import async_fn

        @asynq()
        def capybara():
            val1 = 0
            try:
                val1 += yield async_fn.asynq(1)
            except OverflowError:
                val1 += yield async_fn.asynq(4)
            result(val1)

    @assert_passes()
    def test_no_assignment(self):
        from asynq import asynq, result

        # if the result value isn't assigned, we assume it's a side-effecting operation that can't
        # be batched
        from pyanalyze.tests import async_fn

        @asynq()
        def f():
            yield async_fn.asynq(1)
            yield async_fn.asynq(2)
            # some other code
            val1 = 1
            result(val1)

    @assert_passes()
    def test_augassign_used(self):
        from asynq import asynq

        # if the result value isn't assigned, we assume it's a side-effecting operation that can't
        # be batched
        from pyanalyze.tests import async_fn

        @asynq()
        def f():
            x = 0
            x += yield async_fn.asynq(1)
            yield async_fn.asynq(x)

    @assert_passes()
    def test_combine_in_try(self):
        from asynq import asynq

        from pyanalyze.tests import async_fn

        @asynq()
        def capybara():
            x = yield async_fn.asynq(2)
            try:
                y = yield async_fn.asynq(1)  # E: unnecessary_yield
            except OverflowError:
                y = 3
            return (x, y)

        @asynq()
        def pacarana():
            try:
                y = yield async_fn.asynq(1)
            except OverflowError:
                y = 3
            x = yield async_fn.asynq(2)
            return (x, y)


class TestMissingAsync(TestNameCheckVisitorBase):
    @assert_passes()
    def test_async_method(self):
        from asynq import asynq, result

        class Capybara(object):
            @asynq()
            def eat(self):
                pass

        def fn():
            result((yield Capybara().eat.asynq()))  # E: missing_asynq

    @assert_passes()
    def test_yield_tuple(self):
        from asynq import asynq, result

        @asynq()
        def eat():
            pass

        @asynq()
        def drink():
            pass

        def fn():
            result((yield (eat.asynq(), drink.asynq())))  # E: missing_asynq

    @assert_passes()
    def test_async_function(self):
        from asynq import asynq, result

        @asynq()
        def eat():
            pass

        def fn():
            assert_is_value(eat, KnownValue(eat))
            result((yield eat.asynq()))  # E: missing_asynq

    @assert_passes()
    def test_successful(self):
        from asynq import asynq, result

        @asynq()
        def eat():
            pass

        @asynq()
        def fn():
            result((yield eat.asynq()))

    @assert_passes()
    def test_not_inferred(self):
        def capybara(fn):
            yield fn.asynq()  # E: missing_asynq

        def box_get(box):
            yield box.get_async()  # E: missing_asynq

    def test_autofix(self):
        self.assert_is_changed(
            """
            from asynq import asynq

            def capybara(fn):
                return (yield fn.asynq())
            """,
            """
            from asynq import asynq

            @asynq()
            def capybara(fn):
                return (yield fn.asynq())
            """,
        )
        # make sure it's only added once
        self.assert_is_changed(
            """
            from asynq import asynq

            def capybara(fn, fn2):
                val = yield fn.asynq()
                val2 = yield fn2.asynq()
                return val ^ val2
            """,
            """
            from asynq import asynq

            @asynq()
            def capybara(fn, fn2):
                val = yield fn.asynq()
                val2 = yield fn2.asynq()
                return val ^ val2
            """,
        )

    @assert_passes()
    def test_lambda(self):
        from asynq import asynq
        from asynq.tools import afilter

        @asynq()
        def inner(obj):
            return bool(obj)

        @asynq()
        def outer(objs):
            objs = yield afilter.asynq(
                asynq()(lambda obj: not (yield inner.asynq(obj))), objs
            )
            return objs

    @assert_passes()
    def test_coverage(self):
        from pyanalyze.tests import async_fn

        func = async_fn.asynq

        def capybara(obj: str, unannotated):
            yield obj.title()
            yield unannotated()
            yield func(1)  # E: missing_asynq


class TestDuplicateYield(TestNameCheckVisitorBase):
    @assert_passes()
    def test_duplicate(self):
        from asynq import asynq

        from pyanalyze.tests import async_fn

        @asynq()
        def dupe_none():
            yield None, None  # E: duplicate_yield

        @asynq()
        def extraneous():
            yield None, None  # E: duplicate_yield
            return 3

        @asynq()
        def dupe_call(oid):
            yield async_fn.asynq(oid), async_fn.asynq(oid)  # E: duplicate_yield

        @asynq()
        def way_too_many(oid, oid2):
            yield async_fn.asynq(oid), async_fn.asynq(  # E: duplicate_yield
                oid
            ), async_fn.asynq(oid2)
            len((yield async_fn.asynq(oid), async_fn.asynq(oid)))  # E: duplicate_yield

        @asynq()
        def too_many_assignments():
            a = b = yield None, None  # E: duplicate_yield
            return (a, b)

    @assert_passes()
    def test_not_async(self):
        def normal_generator(lst):
            for uid in lst:
                yield uid, uid

    def test_autofix(self):
        self.assert_is_changed(
            """
            from asynq import asynq

            @asynq()
            def capybara():
                yield None, None
            """,
            """
            from asynq import asynq

            @asynq()
            def capybara():
                yield None
            """,
        )
        self.assert_is_changed(
            """
            from asynq import asynq
            from pyanalyze.tests import async_fn

            @asynq()
            def capybara(oid):
                a, b = yield async_fn.asynq(oid), async_fn.asynq(oid)
            """,
            """
            from asynq import asynq
            from pyanalyze.tests import async_fn

            @asynq()
            def capybara(oid):
                a = yield async_fn.asynq(oid)
                b = a
            """,
        )
        self.assert_is_changed(
            """
            from asynq import asynq
            from pyanalyze.tests import async_fn

            @asynq()
            def capybara(oid):
                a, b, c = yield async_fn.asynq(oid), async_fn.asynq(oid), async_fn.asynq(oid + 1)
            """,
            """
            from asynq import asynq
            from pyanalyze.tests import async_fn

            @asynq()
            def capybara(oid):
                a, c = yield async_fn.asynq(oid), async_fn.asynq(oid + 1)
                b = a
            """,
        )
        self.assert_is_changed(
            """
            from asynq import asynq
            from pyanalyze.tests import async_fn

            @asynq()
            def capybara(oid):
                a, _ = yield async_fn.asynq(oid), async_fn.asynq(oid)
            """,
            """
            from asynq import asynq
            from pyanalyze.tests import async_fn

            @asynq()
            def capybara(oid):
                a = yield async_fn.asynq(oid)
            """,
        )


class TestVarnameGenerator(object):
    def check(self, code, expected_var_name, disallowed_names=set()):
        # the result is Module(body=[Expr(value=<what we want>)])
        node = ast.parse(code).body[0].value
        gen = VarnameGenerator(lambda name: name not in disallowed_names)
        assert expected_var_name == gen.get(node)

    def test_name(self):
        self.check("x", "x_result")
        self.check("CapybaraPower", "capybara_power")

    def test_attr(self):
        self.check("capybara.asynq", "capybara_result")
        self.check("self.capybara.asynq", "capybara")
        self.check("self.render_capybara.asynq", "capybara")
        self.check("self.get_capybara.asynq", "capybara")
        self.check("self._get_capybara", "capybara")
        self.check("capybara.get_async", "capybara_result")

    def test_call(self):
        self.check("capybara.asynq()", "capybara_result")
        self.check("IsGoodCapybara(cid).get_async()", "is_good_capybara")
        self.check("async_call.asynq(self.render_result)", "result")

    def test_ensure_unique(self):
        self.check("x", "x_result2", disallowed_names={"x_result"})
        self.check("x", "x_result3", disallowed_names={"x_result", "x_result2"})

    def test_strip_underscore(self):
        self.check("_capybara", "capybara_result")

    def test_subscript(self):
        self.check("futures[0]", "autogenerated_var")


def test_camel_case_to_snake_case():
    assert "hello" == _camel_case_to_snake_case("hello")
    assert "capybara" == _camel_case_to_snake_case("Capybara")
    assert "capybara_result" == _camel_case_to_snake_case("CapybaraResult")
    assert "http_error" == _camel_case_to_snake_case("HTTPError")
    assert "ñoldor" == _camel_case_to_snake_case("Ñoldor")
    assert "αθήνα" == _camel_case_to_snake_case("Αθήνα")
    assert "ссср_союз" == _camel_case_to_snake_case("СССРСоюз")
