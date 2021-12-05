# static analysis: ignore
from .error_code import ErrorCode
from .name_check_visitor import build_stacked_scopes
from .stacked_scopes import ScopeType, uniq_chain
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    DictIncompleteValue,
    GenericValue,
    KnownValue,
    MultiValuedValue,
    ReferencingValue,
    TypedValue,
    UNINITIALIZED_VALUE,
    assert_is_value,
    make_weak,
)


# just used for its __dict__
class Module(object):
    foo = 1
    bar = None


class TestStackedScopes(object):
    def setup(self):
        self.scopes = build_stacked_scopes(Module)

    def test_scope_type(self):
        assert ScopeType.module_scope == self.scopes.scope_type()

        with self.scopes.add_scope(ScopeType.function_scope, scope_node=None):
            assert ScopeType.function_scope == self.scopes.scope_type()

        assert ScopeType.module_scope == self.scopes.scope_type()

    def test_current_and_module_scope(self):
        assert "foo" in self.scopes.current_scope()
        assert "foo" in self.scopes.module_scope()

        with self.scopes.add_scope(ScopeType.function_scope, scope_node=None):
            assert "foo" not in self.scopes.current_scope()
            assert "foo" in self.scopes.module_scope()

        assert "foo" in self.scopes.current_scope()
        assert "foo" in self.scopes.module_scope()

    def test_get(self):
        assert KnownValue(1) == self.scopes.get("foo", None, None)

        with self.scopes.add_scope(ScopeType.module_scope, scope_node=None):
            self.scopes.set("foo", KnownValue(2), None, None)
            assert KnownValue(2) == self.scopes.get("foo", None, None)

        assert KnownValue(1) == self.scopes.get("foo", None, None)

        assert UNINITIALIZED_VALUE is self.scopes.get("doesnt_exist", None, None)

        # outer class scopes aren't used
        with self.scopes.add_scope(ScopeType.class_scope, scope_node=None):
            self.scopes.set("cls1", KnownValue(1), None, None)
            assert KnownValue(1) == self.scopes.get("cls1", None, None)

            with self.scopes.add_scope(ScopeType.class_scope, scope_node=None):
                self.scopes.set("cls2", KnownValue(1), None, None)
                assert KnownValue(1) == self.scopes.get("cls2", None, None)

                assert UNINITIALIZED_VALUE is self.scopes.get("cls1", None, None)

            assert KnownValue(1) == self.scopes.get("cls1", None, None)

    def test_set(self):
        with self.scopes.add_scope(ScopeType.module_scope, scope_node=None):
            self.scopes.set("multivalue", KnownValue(1), None, None)
            assert KnownValue(1) == self.scopes.get("multivalue", None, None)
            self.scopes.set("multivalue", KnownValue(2), None, None)
            assert MultiValuedValue([KnownValue(1), KnownValue(2)]) == self.scopes.get(
                "multivalue", None, None
            )
            self.scopes.set("multivalue", KnownValue(3), None, None)
            assert MultiValuedValue(
                [KnownValue(1), KnownValue(2), KnownValue(3)]
            ) == self.scopes.get("multivalue", None, None)

            # if the values set are the same, don't make a MultiValuedValue
            self.scopes.set("same", KnownValue(1), None, None)
            assert KnownValue(1) == self.scopes.get("same", None, None)
            self.scopes.set("same", KnownValue(1), None, None)
            assert KnownValue(1) == self.scopes.get("same", None, None)

            # even if they are AnyValue
            any = AnyValue(AnySource.marker)
            self.scopes.set("unresolved", any, None, None)
            assert any is self.scopes.get("unresolved", None, None)
            self.scopes.set("unresolved", any, None, None)
            assert any is self.scopes.get("unresolved", None, None)

    def test_referencing_value(self):
        with self.scopes.add_scope(ScopeType.module_scope, scope_node=None):
            outer = self.scopes.current_scope()
            self.scopes.set("reference", KnownValue(1), None, None)
            multivalue = MultiValuedValue([KnownValue(1), KnownValue(2)])

            with self.scopes.add_scope(ScopeType.module_scope, scope_node=None):
                val = ReferencingValue(outer, "reference")
                self.scopes.set("reference", val, None, None)
                assert KnownValue(1) == self.scopes.get("reference", None, None)
                self.scopes.set("reference", KnownValue(2), None, None)
                assert multivalue == self.scopes.get("reference", None, None)

            assert multivalue == self.scopes.get("reference", None, None)

            self.scopes.set(
                "nonexistent",
                ReferencingValue(self.scopes.module_scope(), "nonexistent"),
                None,
                None,
            )
            assert UNINITIALIZED_VALUE is self.scopes.get("nonexistent", None, None)

            self.scopes.set("is_none", KnownValue(None), None, None)

            with self.scopes.add_scope(ScopeType.function_scope, scope_node=None):
                self.scopes.set(
                    "is_none", ReferencingValue(outer, "is_none"), None, None
                )
                assert AnyValue(AnySource.inference) == self.scopes.get(
                    "is_none", None, None
                )

    def test_typed_value_set(self):
        self.scopes.set("value", TypedValue(dict), None, None)
        assert TypedValue(dict) == self.scopes.get("value", None, None)
        div = DictIncompleteValue(dict, [])  # subclass of TypedValue
        self.scopes.set("value", div, None, None)
        assert div == self.scopes.get("value", None, None)


class TestScoping(TestNameCheckVisitorBase):
    @assert_passes()
    def test_multiple_assignment(self):
        def capybara():
            x = 3
            assert_is_value(x, KnownValue(3))
            x = 4
            assert_is_value(x, KnownValue(4))

    @assert_passes()
    def test_undefined_name(self):
        def capybara():
            return x  # E: undefined_name

    @assert_passes()
    def test_read_before_write(self):
        def capybara():
            print(x)  # E: undefined_name
            x = 3
            print(x)

    @assert_passes()
    def test_function_argument(self):
        def capybara(x):
            assert_is_value(x, AnyValue(AnySource.unannotated))
            x = 3
            assert_is_value(x, KnownValue(3))

    @assert_passes()
    def test_default_arg(self):
        def capybara(x=3):
            assert_is_value(
                x, MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue(3)])
            )

    @assert_passes()
    def test_args_kwargs(self):
        def capybara(*args, **kwargs):
            assert_is_value(args, TypedValue(tuple))
            assert_is_value(
                kwargs,
                GenericValue(dict, [TypedValue(str), AnyValue(AnySource.unannotated)]),
            )

    @assert_passes()
    def test_args_kwargs_annotated(self):
        def capybara(*args: int, **kwargs: int):
            assert_is_value(args, GenericValue(tuple, [TypedValue(int)]))
            assert_is_value(
                kwargs, GenericValue(dict, [TypedValue(str), TypedValue(int)])
            )

    @assert_passes()
    def test_internal_imports(self):
        # nested import froms are tricky because there is no separate AST node for each name, so we
        # need to use a special trick to represent the distinct definition nodes for each name
        import collections

        def capybara():
            from collections import Counter, defaultdict

            assert_is_value(Counter, KnownValue(collections.Counter))
            assert_is_value(defaultdict, KnownValue(collections.defaultdict))

    @assert_passes()
    def test_return_annotation(self):
        import socket

        class Capybara:
            def socket(self) -> socket.error:
                return socket.error()


class TestIf(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        def capybara(cond):
            if cond:
                x = 3
                assert_is_value(x, KnownValue(3))
            else:
                x = 4
                assert_is_value(x, KnownValue(4))
            assert_is_value(x, MultiValuedValue([KnownValue(3), KnownValue(4)]))

    @assert_passes()
    def test_nesting(self):
        def capybara(cond1, cond2):
            if cond1:
                x = 3
                assert_is_value(x, KnownValue(3))
            else:
                if cond2:
                    x = 4
                    assert_is_value(x, KnownValue(4))
                else:
                    x = 5
                    assert_is_value(x, KnownValue(5))
                assert_is_value(x, MultiValuedValue([KnownValue(4), KnownValue(5)]))
            assert_is_value(
                x, MultiValuedValue([KnownValue(3), KnownValue(4), KnownValue(5)])
            )


class TestTry(TestNameCheckVisitorBase):
    @assert_passes(settings={ErrorCode.possibly_undefined_name: False})
    def test_except(self):
        def capybara():
            try:
                x = 3
                assert_is_value(x, KnownValue(3))
            except NameError as e:
                assert_is_value(e, TypedValue(NameError))
                x = 4
                assert_is_value(x, KnownValue(4))
            except (RuntimeError, ValueError) as e:
                assert_is_value(
                    e,
                    MultiValuedValue(
                        [TypedValue(RuntimeError), TypedValue(ValueError)]
                    ),
                )
            assert_is_value(
                x,
                MultiValuedValue(
                    [KnownValue(3), KnownValue(4), AnyValue(AnySource.error)]
                ),
            )

    @assert_passes()
    def test_set_before_try(self):
        def capybara():
            x = 1
            try:
                x = 2
                assert_is_value(x, KnownValue(2))
            except NameError:
                assert_is_value(x, MultiValuedValue([KnownValue(1), KnownValue(2)]))
                x = 3
                assert_is_value(x, KnownValue(3))
            except RuntimeError:
                assert_is_value(x, MultiValuedValue([KnownValue(1), KnownValue(2)]))
                x = 4
                assert_is_value(x, KnownValue(4))
            assert_is_value(
                x, MultiValuedValue([KnownValue(2), KnownValue(3), KnownValue(4)])
            )

    @assert_passes()
    def test_multiple_except(self):
        def capybara():
            try:
                x = 3
                assert_is_value(x, KnownValue(3))
            except NameError:
                x = 4
                assert_is_value(x, KnownValue(4))
            except IOError:
                x = 5
                assert_is_value(x, KnownValue(5))
            assert_is_value(
                x, MultiValuedValue([KnownValue(3), KnownValue(4), KnownValue(5)])
            )

    @assert_passes()
    def test_else(self):
        def capybara():
            try:
                x = 3
                assert_is_value(x, KnownValue(3))
            except NameError:
                x = 4
                assert_is_value(x, KnownValue(4))
            else:
                x = 5
                assert_is_value(x, KnownValue(5))
            assert_is_value(x, MultiValuedValue([KnownValue(5), KnownValue(4)]))

    @assert_passes()
    def test_finally(self):
        def capybara():
            try:
                x = 3
                assert_is_value(x, KnownValue(3))
            finally:
                x = 4
                assert_is_value(x, KnownValue(4))
            assert_is_value(x, KnownValue(4))

    @assert_passes()
    def test_finally_regression():
        import subprocess

        def test_full():
            clients = []
            try:
                clients.append(subprocess.Popen([]))
            finally:
                for client in clients:
                    client.kill()

    @assert_passes()
    def test_finally_plus_if(self):
        # here an approach that simply ignores the assignments in the try block while examining the
        # finally block would fail
        def capybara():
            x = 0
            assert_is_value(x, KnownValue(0))
            try:
                x = 1
                assert_is_value(x, KnownValue(1))
            finally:
                assert_is_value(x, MultiValuedValue([KnownValue(0), KnownValue(1)]))

    @assert_passes()
    def test_finally_plus_return(self):
        def capybara():
            x = 0
            assert_is_value(x, KnownValue(0))
            try:
                x = 1
                assert_is_value(x, KnownValue(1))
                return
            finally:
                assert_is_value(x, MultiValuedValue([KnownValue(0), KnownValue(1)]))

    @assert_passes()
    def test_bad_except_handler(self):
        def capybara():
            try:
                x = 1
            except 42 as fortytwo:  # E: bad_except_handler
                print(fortytwo)
            else:
                print(x)


class TestLoops(TestNameCheckVisitorBase):
    @assert_passes(settings={ErrorCode.possibly_undefined_name: False})
    def test_conditional_in_loop(self):
        def capybara():
            for i in range(2):
                if i == 1:
                    print(x)
                    assert_is_value(
                        x, MultiValuedValue([AnyValue(AnySource.error), KnownValue(3)])
                    )
                else:
                    x = 3
                    assert_is_value(x, KnownValue(3))
            assert_is_value(
                x, MultiValuedValue([AnyValue(AnySource.error), KnownValue(3)])
            )

    @assert_passes()
    def test_second_assignment_in_loop(self):
        def capybara():
            hide_until = None
            for _ in range(3):
                assert_is_value(
                    hide_until, MultiValuedValue([KnownValue(None), KnownValue((1, 2))])
                )
                if hide_until:
                    print(hide_until[1])
                hide_until = (1, 2)

    @assert_passes()
    def test_for_else(self):
        def capybara():
            for _ in range(2):
                x = 3
                assert_is_value(x, KnownValue(3))
            else:
                x = 4
                assert_is_value(x, KnownValue(4))
            assert_is_value(x, MultiValuedValue([KnownValue(3), KnownValue(4)]))

    @assert_passes()
    def test_for_always_entered(self):
        def capybara():
            x = 3
            assert_is_value(x, KnownValue(3))
            for _ in [0, 1]:
                x = 4
                assert_is_value(x, KnownValue(4))
            assert_is_value(x, KnownValue(4))

    @assert_passes()
    def test_range_always_entered(self):
        def capybara():
            for i in range(2):
                assert_is_value(i, KnownValue(0) | KnownValue(1))
            assert_is_value(i, KnownValue(0) | KnownValue(1))

    @assert_passes(settings={ErrorCode.possibly_undefined_name: False})
    def test_use_after_for(self):
        def capybara(x):
            for _ in range(x):
                y = 4
                break

            assert_is_value(
                y, MultiValuedValue([KnownValue(4), AnyValue(AnySource.error)])
            )

    @assert_passes(settings={ErrorCode.possibly_undefined_name: False})
    def test_use_after_for_conditional(self):
        def capybara(x):
            for _ in range(2):
                if x > 2:
                    y = 4
                    break

            assert_is_value(
                y, MultiValuedValue([KnownValue(4), AnyValue(AnySource.error)])
            )

    @assert_passes(settings={ErrorCode.possibly_undefined_name: False})
    def test_while(self):
        def capybara():
            while bool():
                x = 3
                assert_is_value(x, KnownValue(3))
            assert_is_value(
                x, MultiValuedValue([AnyValue(AnySource.error), KnownValue(3)])
            )

    @assert_passes()
    def test_while_always_entered(self):
        def capybara():
            while True:
                x = 3
                assert_is_value(x, KnownValue(3))
                break
            assert_is_value(x, KnownValue(3))

    @assert_passes()
    def test_while_else(self):
        def capybara():
            while bool():
                x = 3
                assert_is_value(x, KnownValue(3))
            else:
                x = 4
                assert_is_value(x, KnownValue(4))
            assert_is_value(x, MultiValuedValue([KnownValue(3), KnownValue(4)]))

    @assert_passes()
    def test_recursive_func_in_loop(self):
        def capybara(xs):
            for x in xs:

                def do_something(y):
                    if x:
                        do_something(y)

                do_something(x)


class TestUnusedVariable(TestNameCheckVisitorBase):
    @assert_passes()
    def test_used(self):
        def capybara(condition):
            y = 3
            print(y)

            z = 3

            def nested():
                print(z)

            x = 4
            if condition:
                print(x)

    @assert_passes()
    def test_unused(self):
        def capybara():
            y = 3  # E: unused_variable

    def test_replacement(self):
        self.assert_is_changed(
            """
def capybara():
    y = 3
    return 3
""",
            """
def capybara():
    return 3
""",
        )

    @assert_passes()
    def test_unused_then_used(self):
        def capybara():
            y = 3  # E: unused_variable
            y = 4
            return y

    @assert_passes()
    def test_unused_in_if(self):
        def capybara(condition):
            if condition:
                x = 3  # E: unused_variable
            x = 4
            return x

    @assert_passes()
    def test_while_loop(self):
        def capybara(condition):
            rlist = condition()
            while rlist:
                rlist = condition()

            num_items = 0
            while num_items < 10:
                if condition:
                    num_items += 1

    @assert_passes(settings={ErrorCode.use_fstrings: False})
    def test_try_finally(self):
        def func():
            return 1

        def capybara():
            x = 0

            try:
                x = func()
            finally:
                print("%d" % x)  # x is a number

    @assert_passes()
    def test_for_may_not_run(self):
        def capybara(iterable):
            # this is not unused, because iterable may be empty
            x = 0
            for x in iterable:
                print(x)
                break
            print(x)

    @assert_passes()
    def test_nesting(self):
        def capybara():
            def inner():
                print(x)

            x = 3
            inner()
            x = 4


class TestUnusedVariableComprehension(TestNameCheckVisitorBase):
    @assert_passes()
    def test_comprehension(self):
        def single_unused():
            return [None for i in range(10)]  # E: unused_variable

        def used():
            return [i for i in range(10)]

        def both_unused(pairs):
            return [None for a, b in pairs]  # E: unused_variable  # E: unused_variable

        def capybara(pairs):
            # this is OK; in real code the name of "b" might serve as useful documentation about
            # what is in "pairs"
            return [a for a, b in pairs]

    def test_replacement(self):
        self.assert_is_changed(
            """
def capybara():
    return [None for i in range(10)]
""",
            """
def capybara():
    return [None for _ in range(10)]
""",
        )


class TestUnusedVariableUnpacking(TestNameCheckVisitorBase):
    @assert_passes()
    def test_unused_in_yield(self):
        from asynq import asynq, result

        @asynq()
        def kerodon(i):
            return i

        @asynq()
        def capybara():
            a, b = yield kerodon.asynq(1), kerodon.asynq(2)  # E: unused_variable
            result(a)

    @assert_passes()
    def test_async_returns_pair(self):
        from asynq import asynq, result

        @asynq()
        def returns_pair():
            return 1, 2

        @asynq()
        def capybara():
            a, b = yield returns_pair.asynq()
            result(a)

    @assert_passes()
    def test_all_unused(self):
        def capybara(pair):
            a, b = pair  # E: unused_variable  # E: unused_variable

    @assert_passes()
    def test_some_used(self):
        def capybara(pair):
            a, b = pair
            return a

    @assert_passes()
    def test_multiple_assignment(self):
        def capybara(pair):
            c = a, b = pair  # E: unused_variable  # E: unused_variable
            return c

    @assert_passes()
    def test_used_in_multiple_assignment(self):
        def capybara(pair):
            a, b = c, d = pair
            return a + d

    @assert_passes()
    def test_nested_unpack(self):
        def capybara(obj):
            (a, b), c = obj
            return c

    @assert_passes()
    def test_used_in_annassign(self):
        def capybara(condition):
            x: int
            if condition:
                x = 1
            else:
                x = 2
            return x


class TestLeavesScope(TestNameCheckVisitorBase):
    @assert_passes()
    def test_leaves_scope(self):
        def capybara(cond):
            if cond:
                return
            else:
                x = 3

            print(x)

    @assert_passes()
    def test_try_always_leaves_scope(self):
        def capybara(cond):
            try:
                x = 3
            except ValueError:
                if cond:
                    raise
                else:
                    return None

            print(x)

    @assert_passes()
    def test_try_may_leave_scope(self):
        def capybara(cond):
            try:
                x = 3
            except ValueError:
                if cond:
                    pass
                else:
                    return None

            print(x)  # E: possibly_undefined_name

    @assert_passes()
    def test_assert_false(self):
        def capybara(cond):
            if cond:
                assert False
            else:
                x = 3

            print(x)

    @assert_passes()
    def test_after_assert_false(self):
        def capybara(cond):
            assert False
            if cond:
                x = True
            else:
                # For some reason in Python 2.7, False gets inferred as Any
                # after the assert False, but True and None still work.
                x = None
            y = None
            assert_is_value(y, KnownValue(None))
            assert_is_value(x, MultiValuedValue([KnownValue(True), KnownValue(None)]))

    @assert_passes()
    def test_elif_assert_false(self):
        def capybara(cond):
            if cond == 1:
                x = 3
            elif cond == 2:
                x = 4
            else:
                assert 0

            print(x)

    @assert_passes()
    def test_visit_assert_message(self):
        from typing import Union

        def needs_int(x: int) -> None:
            pass

        def capybara(x: Union[int, str]) -> None:
            assert_is_value(x, MultiValuedValue([TypedValue(int), TypedValue(str)]))

            assert isinstance(x, str), needs_int(x)
            assert_is_value(x, TypedValue(str))

    @assert_passes()
    def test_no_cross_function_propagation(self):
        def capybara(cond):
            if cond == 1:
                x = 3
            else:
                pass

            return x  # static analysis: ignore[possibly_undefined_name]

        def kerodon():
            # make sure we don't propagate the UNINITIALIZED_VALUE from
            # inside capybara() to here
            y = capybara(2)
            print(y)


class TestConstraints(TestNameCheckVisitorBase):
    @assert_passes()
    def test_assert_truthy(self):
        def capybara(x):
            if x:
                y = True
            else:
                y = False
            assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))
            assert y
            assert_is_value(y, KnownValue(True))

    @assert_passes()
    def test_assert_falsy(self):
        def capybara(x):
            if x:
                y = True
            else:
                y = False
            assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))
            assert not y
            assert_is_value(y, KnownValue(False))

    @assert_passes()
    def test_no_constraints_from_branches(self):
        def capybara(x):
            if x:
                y = True
            else:
                y = False

            if x:
                assert_is_value(
                    y, MultiValuedValue([KnownValue(True), KnownValue(False)])
                )
                assert y
                assert_is_value(y, KnownValue(True))
            # Constraints do not survive past the if block.
            assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))

    @assert_passes()
    def test_if(self):
        def capybara(x):
            if x:
                y = True
            else:
                y = False

            assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))
            if y:
                assert_is_value(y, KnownValue(True))
            else:
                assert_is_value(y, KnownValue(False))
            assert_is_value(y, KnownValue(True)) if y else assert_is_value(
                y, KnownValue(False)
            )

    @assert_passes()
    def test_isinstance(self):
        class A(object):
            pass

        class B(A):
            pass

        class C(A):
            pass

        def capybara(x):
            assert_is_value(x, AnyValue(AnySource.unannotated))
            if isinstance(x, int):
                assert_is_value(x, TypedValue(int))
            else:
                assert_is_value(x, AnyValue(AnySource.unannotated))

            if isinstance(x, A):
                assert_is_value(x, TypedValue(A))
                if isinstance(x, B):
                    assert_is_value(x, TypedValue(B))
                    if isinstance(x, C):
                        # Incompatible constraints result in Any.
                        assert_is_value(x, AnyValue(AnySource.unreachable))
            if isinstance(x, B):
                assert_is_value(x, TypedValue(B))
                if isinstance(x, A):
                    # Less precise constraints are ignored.
                    assert_is_value(x, TypedValue(B))

            x = B()
            assert_is_value(x, TypedValue(B))
            if isinstance(x, A):
                # Don't widen the type to A.
                assert_is_value(x, TypedValue(B))

        def kerodon(cond1, cond2, val, lst: list):
            if cond1:
                x = int(val)
            elif cond2:
                x = str(val)
            else:
                x = lst
            assert_is_value(
                x,
                MultiValuedValue([TypedValue(int), TypedValue(str), TypedValue(list)]),
            )

            if isinstance(x, (int, str)):
                assert_is_value(x, MultiValuedValue([TypedValue(int), TypedValue(str)]))
            else:
                assert_is_value(x, TypedValue(list))

            assert_is_value(
                x,
                MultiValuedValue([TypedValue(int), TypedValue(str), TypedValue(list)]),
            )
            if isinstance(x, int) or isinstance(x, str):
                assert_is_value(x, MultiValuedValue([TypedValue(int), TypedValue(str)]))
            else:
                assert_is_value(x, TypedValue(list))

        def paca(cond1, cond2):
            if cond1:
                x = True
            elif cond2:
                x = False
            else:
                x = None

            if (x is not True and x is not False) or (x is True):
                assert_is_value(
                    x, MultiValuedValue([KnownValue(None), KnownValue(True)])
                )
            else:
                assert_is_value(x, KnownValue(False))

    @assert_passes()
    def test_qcore_asserts(self):
        from qcore.asserts import assert_is, assert_is_not, assert_is_instance

        def capybara(cond):
            if cond:
                x = True
                y = True
            else:
                x = False
                y = False

            assert_is_value(x, MultiValuedValue([KnownValue(True), KnownValue(False)]))
            assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))
            assert x is True
            assert True is y
            assert_is_value(x, KnownValue(True))
            assert_is_value(y, KnownValue(True))

        def paca(cond):
            if cond:
                x = True
                y = True
            else:
                x = False
                y = False

            assert_is_value(x, MultiValuedValue([KnownValue(True), KnownValue(False)]))
            assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))
            assert x is not True
            assert True is not y
            assert_is_value(x, KnownValue(False))
            assert_is_value(y, KnownValue(False))

        def mara(cond, cond2):
            assert_is_value(cond, AnyValue(AnySource.unannotated))
            assert_is_instance(cond, int)
            assert_is_value(cond, TypedValue(int))

            assert_is_instance(cond2, (int, str))
            assert_is_value(cond2, MultiValuedValue([TypedValue(int), TypedValue(str)]))

    @assert_passes()
    def test_is_or_is_not(self):
        def capybara(x):
            if x:
                y = True
            else:
                y = False

            assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))
            if y is True:
                assert_is_value(y, KnownValue(True))
            else:
                assert_is_value(y, KnownValue(False))
            if y is not True:
                assert_is_value(y, KnownValue(False))
            else:
                assert_is_value(y, KnownValue(True))

    @assert_passes()
    def test_and_or(self):
        true_or_false = MultiValuedValue([KnownValue(True), KnownValue(False)])

        def capybara(x, y):
            if x is True and y is True:
                assert_is_value(x, KnownValue(True))
                assert_is_value(y, KnownValue(True))
            else:
                # no constraints from the inverse of an AND constraint
                assert_is_value(x, AnyValue(AnySource.unannotated))
                assert_is_value(y, AnyValue(AnySource.unannotated))

        def kerodon(x):
            if x is True and assert_is_value(x, KnownValue(True)):
                pass
            # After the if it's either True (if the if branch was taken)
            # or Any (if it wasn't). This is not especially
            # useful in this case, but hopefully harmless.
            assert_is_value(
                x, MultiValuedValue([KnownValue(True), AnyValue(AnySource.unannotated)])
            )

        def paca(x):
            if x:
                y = True
                z = True
            else:
                y = False
                z = False

            if y is True or z is True:
                assert_is_value(y, true_or_false)
                assert_is_value(z, true_or_false)
            else:
                assert_is_value(y, KnownValue(False))
                assert_is_value(z, KnownValue(False))

        def pacarana(x):
            # OR constraints within the conditional
            if x:
                z = True
            else:
                z = False
            if z is True or assert_is_value(z, KnownValue(False)):
                pass

        def hutia(x):
            if x:
                y = True
            else:
                y = False

            if x and y:
                assert_is_value(y, KnownValue(True))
            else:
                assert_is_value(y, true_or_false)

        def mara(x):
            if x:
                y = True
                z = True
            else:
                y = False
                z = False

            if not (y is True and z is True):
                assert_is_value(y, true_or_false)
                assert_is_value(z, true_or_false)
            else:
                assert_is_value(y, KnownValue(True))
                assert_is_value(z, KnownValue(True))

        def phoberomys(cond):
            if cond:
                x = True
                y = True
                z = True
            else:
                x = False
                y = False
                z = False

            if not ((x is False or y is False) or z is True):
                assert_is_value(x, KnownValue(True))
                assert_is_value(y, KnownValue(True))
                assert_is_value(z, KnownValue(False))
            else:
                assert_is_value(x, true_or_false)
                assert_is_value(y, true_or_false)
                assert_is_value(z, true_or_false)

        def llitun(cond):
            if cond:
                x = True
                y = True
                z = True
            else:
                x = False
                y = False
                z = False
            if x and y and z:
                assert_is_value(x, KnownValue(True))
                assert_is_value(y, KnownValue(True))
                assert_is_value(z, KnownValue(True))
            else:
                assert_is_value(x, true_or_false)
                assert_is_value(y, true_or_false)
                assert_is_value(z, true_or_false)

        def coypu(cond):
            if cond:
                x = True
                y = True
                z = True
            else:
                x = False
                y = False
                z = False
            if x or y or z:
                assert_is_value(x, true_or_false)
                assert_is_value(y, true_or_false)
                assert_is_value(z, true_or_false)
            else:
                assert_is_value(x, KnownValue(False))
                assert_is_value(y, KnownValue(False))
                assert_is_value(z, KnownValue(False))

    @assert_passes()
    def test_set_in_condition(self):
        def capybara(x):
            if x:
                y = True
            else:
                y = False
            assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))
            if not y:
                assert_is_value(y, KnownValue(False))
                y = True
            assert_is_value(y, KnownValue(True))

    @assert_passes()
    def test_optional_becomes_non_optional(self):
        from typing import Optional

        def capybara(x: Optional[int]) -> None:
            assert_is_value(x, MultiValuedValue([TypedValue(int), KnownValue(None)]))
            if not x:
                x = int(0)
            assert_is_value(x, TypedValue(int))

    @assert_passes()
    def test_reset_on_assignment(self):
        def capybara(x):
            if x:
                y = True
            else:
                y = False
            if y is True:
                assert_is_value(y, KnownValue(True))
                y = bool(x)
                assert_is_value(y, TypedValue(bool))

    @assert_passes()
    def test_constraint_on_arg_type(self):
        from typing import Optional

        def kerodon() -> Optional[int]:
            return 3

        def capybara() -> None:
            x = kerodon()
            assert_is_value(x, MultiValuedValue([TypedValue(int), KnownValue(None)]))

            if x:
                assert_is_value(x, TypedValue(int))
            else:
                assert_is_value(
                    x, MultiValuedValue([TypedValue(int), KnownValue(None)])
                )
            if x is not None:
                assert_is_value(x, TypedValue(int))
            else:
                assert_is_value(x, KnownValue(None))

    @assert_passes()
    def test_constraint_in_nested_scope(self):
        from typing import Optional

        def capybara(x: Optional[int], z):
            if x is None:
                return

            assert_is_value(x, TypedValue(int))

            def nested():
                assert_is_value(x, TypedValue(int))

            return [assert_is_value(x, TypedValue(int)) for _ in z]

    @assert_passes()
    def test_repeated_constraints(self):
        def capybara(cond):
            if cond:
                x = True
            else:
                x = False
            assert_is_value(x, MultiValuedValue([KnownValue(True), KnownValue(False)]))

            # Tests that this completes in a reasonable time.
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            assert_is_value(x, MultiValuedValue([KnownValue(True), KnownValue(False)]))

    @assert_passes()
    def test_nonlocal_unresolved(self):
        def capybara(x):
            def nested():
                while True:
                    assert_is_value(x, AnyValue(AnySource.unannotated))
                    if x:
                        pass

            return nested()

    @assert_passes()
    def test_nonlocal_unresolved_if(self):
        def capybara(x):
            def nested():
                assert_is_value(x, AnyValue(AnySource.unannotated))
                if x:
                    assert_is_value(x, AnyValue(AnySource.unannotated))

            return nested()

    @assert_passes()
    def test_nonlocal_known(self):
        def capybara(y):
            if y:
                x = True
            else:
                x = False

            def nested():
                assert_is_value(
                    x, MultiValuedValue([KnownValue(True), KnownValue(False)])
                )
                if x:
                    assert_is_value(x, KnownValue(True))
                else:
                    assert_is_value(x, KnownValue(False))

    @assert_passes()
    def test_nonlocal_known_with_write(self):
        def capybara(y):
            if y:
                x = True
            else:
                x = False

            def nested():
                nonlocal x
                assert_is_value(
                    x, MultiValuedValue([KnownValue(True), KnownValue(False)])
                )
                if x:
                    assert_is_value(x, KnownValue(True))
                else:
                    assert_is_value(x, KnownValue(False))
                    x = True
                    assert_is_value(x, KnownValue(True))

    @assert_passes()
    def test_nonlocal_in_loop(self):
        def capybara(x):
            def nested(y):
                for _ in y:
                    if x:
                        pass

    @assert_passes()
    def test_nonlocal_not_unused(self):
        def _get_call_point(x, y):
            frame = x
            while y(frame):
                frame = frame.f_back
            return {"filename": frame.f_code.co_filename, "line_no": frame.f_lineno}

    @assert_passes()
    def test_conditional_assignment_to_global(self):
        _disk_size_with_low_usage = 0

        def _report_boxes_with_low_disk_usage(tier):
            global _disk_size_with_low_usage
            x = 0
            if tier.startswith("lego"):
                _disk_size_with_low_usage = 3
            x += _disk_size_with_low_usage
            _disk_size_with_low_usage = 0
            return x

    @assert_passes()
    def test_comprehension(self):
        def maybe_int(x):
            if x:
                return int(x)
            else:
                return None

        def capybara(x, y):
            assert_is_value(
                maybe_int(x), MultiValuedValue([TypedValue(int), KnownValue(None)])
            )

            lst = [maybe_int(elt) for elt in y]
            assert_is_value(
                lst,
                make_weak(
                    GenericValue(
                        list, [MultiValuedValue([TypedValue(int), KnownValue(None)])]
                    )
                ),
            )
            lst2 = [elt for elt in lst if elt]
            assert_is_value(lst2, make_weak(GenericValue(list, [TypedValue(int)])))

    @assert_passes()
    def test_comprehension_composite(self):
        from dataclasses import dataclass
        from typing import Optional, Tuple, List

        @dataclass
        class Capybara:
            x: Optional[int]

        def use_attr(c: List[Capybara]) -> None:
            assert_is_value(
                [elt.x for elt in c],
                make_weak(
                    GenericValue(
                        list, [MultiValuedValue([TypedValue(int), KnownValue(None)])]
                    )
                ),
            )
            assert_is_value(
                [elt.x for elt in c if elt.x is not None],
                make_weak(GenericValue(list, [TypedValue(int)])),
            )
            assert_is_value(
                [elt.x for elt in c if elt.x],
                make_weak(GenericValue(list, [TypedValue(int)])),
            )

        def use_subscript(d: List[Tuple[int, Optional[int]]]) -> None:
            assert_is_value(
                [pair[1] for pair in d],
                make_weak(
                    GenericValue(
                        list, [MultiValuedValue([TypedValue(int), KnownValue(None)])]
                    )
                ),
            )
            assert_is_value(
                [pair[1] for pair in d if pair[1] is not None],
                make_weak(GenericValue(list, [TypedValue(int)])),
            )
            assert_is_value(
                [pair[1] for pair in d if pair[1]],
                make_weak(GenericValue(list, [TypedValue(int)])),
            )

    @assert_passes()
    def test_while(self):
        def capybara(x):
            if x:
                y = True
            else:
                y = False
            assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))
            while y:
                assert_is_value(y, KnownValue(True))
            assert_is_value(y, MultiValuedValue([KnownValue(True), KnownValue(False)]))

    @assert_passes()
    def test_while_hasattr(self):
        from typing import Optional
        from pyanalyze.value import HasAttrExtension

        def capybara(x: Optional[int]):
            assert_is_value(x, MultiValuedValue([KnownValue(None), TypedValue(int)]))
            while x is not None and hasattr(x, "name"):
                assert_is_value(
                    x,
                    AnnotatedValue(
                        TypedValue(int),
                        [
                            HasAttrExtension(
                                KnownValue("name"), AnyValue(AnySource.inference)
                            )
                        ],
                    ),
                )

    @assert_passes()
    def test_unconstrained_composite(self):
        class Foo(object):
            def has_images(self):
                pass

        class InlineEditor:
            def init(self, input, is_qtext=False):
                if is_qtext:
                    value = input
                else:
                    value = ""

                assert_is_value(
                    value,
                    MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue("")]),
                )

                self.value = value

                assert_is_value(
                    self.value,
                    MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue("")]),
                )

            def tree(self):
                assert_is_value(
                    self.value,
                    MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue("")]),
                )
                if isinstance(self.value, Foo) and self.value.has_images():
                    assert_is_value(self.value, TypedValue(Foo))
                else:
                    assert_is_value(
                        self.value,
                        MultiValuedValue(
                            [AnyValue(AnySource.unannotated), KnownValue("")]
                        ),
                    )
                assert_is_value(
                    self.value,
                    MultiValuedValue(
                        [
                            TypedValue(Foo),
                            AnyValue(AnySource.unannotated),
                            KnownValue(""),
                        ]
                    ),
                )

    @assert_passes()
    def test_operator_constraints(self):
        from typing import Union
        from typing_extensions import Literal

        container = {1, 2, 3}

        def capybara(cond):
            x = 1 if cond else 2
            assert_is_value(x, MultiValuedValue([KnownValue(1), KnownValue(2)]))
            if x == 1:
                assert_is_value(x, KnownValue(1))
            else:
                assert_is_value(x, KnownValue(2))
            if x in (2,):
                assert_is_value(x, KnownValue(2))
            else:
                assert_is_value(x, KnownValue(1))
            if "x" in cond:
                assert_is_value(cond, AnyValue(AnySource.unannotated))

        def pacarana(x: Union[Literal["x"], int]):
            assert_is_value(x, KnownValue("x") | TypedValue(int))
            if x == 0:
                assert_is_value(x, KnownValue(0))
            elif x == "x":
                assert_is_value(x, KnownValue("x"))
            else:
                assert_is_value(x, TypedValue(int))

        def moco(x: Union[Literal["x"], int]):
            assert_is_value(x, KnownValue("x") | TypedValue(int))
            if x != 0:
                assert_is_value(x, KnownValue("x") | TypedValue(int))
            else:
                assert_is_value(x, KnownValue(0))

        def hutia(x: str, y: object):
            if x in ["a", "b"]:
                assert_is_value(x, KnownValue("a") | KnownValue("b"))
            if y in container:
                assert_is_value(y, KnownValue(1) | KnownValue(2) | KnownValue(3))

    @assert_passes()
    def test_preserve_annotated(self):
        from typing_extensions import Annotated
        from typing import Optional

        AnnotatedUnion = AnnotatedValue(
            TypedValue(str), [KnownValue(1)]
        ) | AnnotatedValue(KnownValue(None), [KnownValue(1)])

        def capybara(x: Annotated[Optional[str], 1]) -> None:
            assert_is_value(x, AnnotatedUnion)

            if x:
                assert_is_value(x, AnnotatedValue(TypedValue(str), [KnownValue(1)]))
            else:
                # None or the empty string
                assert_is_value(x, AnnotatedUnion)

        def pacarana(x: Annotated[Optional[str], 1]) -> None:
            assert_is_value(x, AnnotatedUnion)
            if x is not None:
                assert_is_value(x, AnnotatedValue(TypedValue(str), [KnownValue(1)]))
            else:
                assert_is_value(x, AnnotatedValue(KnownValue(None), [KnownValue(1)]))

        def agouti(x: Annotated[Optional[str], 1]) -> None:
            assert_is_value(x, AnnotatedUnion)
            if isinstance(x, str):
                assert_is_value(x, AnnotatedValue(TypedValue(str), [KnownValue(1)]))
            else:
                assert_is_value(x, AnnotatedValue(KnownValue(None), [KnownValue(1)]))


class TestComposite(TestNameCheckVisitorBase):
    @assert_passes()
    def test_assignment(self):
        class Capybara(object):
            def __init__(self, x):
                self.x = x

            def eat(self):
                assert_is_value(
                    self.x,
                    MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue(1)]),
                )
                self.x = 1
                assert_is_value(self.x, KnownValue(1))

                self = Capybara(2)
                assert_is_value(
                    self.x,
                    MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue(1)]),
                )

    @assert_passes()
    def test_conditional_attribute_assign(self):
        class Capybara(object):
            def __init__(self, x):
                self.x = int(x)

            def eat(self, cond, val):
                if cond:
                    self.x = int(val)
                x = self.x
                assert_is_value(x, TypedValue(int))

    @assert_passes()
    def test_constraint(self):
        class Capybara(object):
            def __init__(self, x):
                self.x = x

            def eat(self, val):
                self.x = val
                if isinstance(self.x, int):
                    assert_is_value(self.x, TypedValue(int))

            def eat_no_assign(self):
                if isinstance(self.x, int):
                    assert_is_value(self.x, TypedValue(int))

    @assert_passes()
    def test_subscript(self):
        from typing import Any, Dict

        def capybara(x: Dict[str, Any]) -> None:
            assert_is_value(x["a"], AnyValue(AnySource.explicit))
            x["a"] = 1
            assert_is_value(x["a"], KnownValue(1))
            if isinstance(x["c"], int):
                assert_is_value(x["c"], TypedValue(int))
            if x["b"] is None:
                assert_is_value(x["b"], KnownValue(None))

    @assert_passes()
    def test_unhashable_subscript(self):
        def capybara(df):
            # make sure this doesn't crash
            df[["a", "b"]] = 42
            print(df[["a", "b"]])


def test_uniq_chain():
    assert [] == uniq_chain([])
    assert list(range(3)) == uniq_chain(range(3) for _ in range(3))
    assert [1] == uniq_chain([1, 1, 1] for _ in range(3))
