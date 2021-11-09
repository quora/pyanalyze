# static analysis: ignore
from pyanalyze.implementation import assert_is_value
from collections.abc import Sequence
from qcore.asserts import assert_eq

from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CanAssignError,
    GenericValue,
    KnownValue,
    MultiValuedValue,
    SequenceIncompleteValue,
    TypedDictValue,
    TypedValue,
)
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_fails, assert_passes, skip_before
from .error_code import ErrorCode
from .signature import Signature, SigParameter as P
from .test_value import CTX

TupleInt = GenericValue(tuple, [TypedValue(int)])
TupleBool = GenericValue(tuple, [TypedValue(bool)])
TupleObject = GenericValue(tuple, [TypedValue(object)])
DictInt = GenericValue(dict, [TypedValue(str), TypedValue(int)])
DictBool = GenericValue(dict, [TypedValue(str), TypedValue(bool)])
DictObject = GenericValue(dict, [TypedValue(str), TypedValue(object)])


def test_stringify() -> None:
    assert_eq("() -> Any[unannotated]", str(Signature.make([])))
    assert_eq("() -> int", str(Signature.make([], TypedValue(int))))
    assert_eq(
        "@asynq () -> int", str(Signature.make([], TypedValue(int), is_asynq=True))
    )
    assert_eq(
        "(...) -> int", str(Signature.make([], TypedValue(int), is_ellipsis_args=True))
    )
    assert_eq(
        "(x: int) -> int",
        str(Signature.make([P("x", annotation=TypedValue(int))], TypedValue(int))),
    )


class TestCanAssign:
    def can(self, left: Signature, right: Signature) -> None:
        tv_map = left.can_assign(right, CTX)
        assert isinstance(
            tv_map, dict
        ), f"cannot assign {right} to {left} due to {tv_map}"

    def cannot(self, left: Signature, right: Signature) -> None:
        tv_map = left.can_assign(right, CTX)
        assert isinstance(tv_map, CanAssignError), f"can assign {right} to {left}"

    def test_return_value(self) -> None:
        self.can(
            Signature.make([], AnyValue(AnySource.marker)),
            Signature.make([], TypedValue(int)),
        )
        self.can(
            Signature.make([], TypedValue(int)), Signature.make([], TypedValue(int))
        )
        self.can(
            Signature.make([], TypedValue(int)), Signature.make([], TypedValue(bool))
        )
        self.cannot(
            Signature.make([], TypedValue(bool)), Signature.make([], TypedValue(int))
        )

    def test_pos_only(self):
        pos_only_int = P("x", annotation=TypedValue(int), kind=P.POSITIONAL_ONLY)
        pos_only_object = P("y", annotation=TypedValue(object), kind=P.POSITIONAL_ONLY)
        pos_only_bool = P("z", annotation=TypedValue(bool), kind=P.POSITIONAL_ONLY)
        pos_kw_int = P("a", annotation=TypedValue(int), kind=P.POSITIONAL_OR_KEYWORD)
        pos_only_sig = Signature.make([pos_only_int])
        self.can(pos_only_sig, pos_only_sig)
        self.can(pos_only_sig, Signature.make([pos_kw_int]))
        self.can(pos_only_sig, Signature.make([pos_only_object]))
        self.cannot(pos_only_sig, Signature.make([pos_only_bool]))
        self.cannot(
            pos_only_sig,
            Signature.make([P("x", annotation=TypedValue(int), kind=P.KEYWORD_ONLY)]),
        )

        # *args interaction
        self.can(
            pos_only_sig,
            Signature.make([P("whatever", annotation=TupleInt, kind=P.VAR_POSITIONAL)]),
        )
        self.cannot(
            pos_only_sig,
            Signature.make([P("x", annotation=TupleBool, kind=P.VAR_POSITIONAL)]),
        )

    def test_pos_or_keyword(self) -> None:
        pos_kw_int = P("a", annotation=TypedValue(int), kind=P.POSITIONAL_OR_KEYWORD)
        pos_kw_int_b = P("b", annotation=TypedValue(int), kind=P.POSITIONAL_OR_KEYWORD)
        pos_kw_object = P(
            "a", annotation=TypedValue(object), kind=P.POSITIONAL_OR_KEYWORD
        )
        pos_kw_bool = P("a", annotation=TypedValue(bool), kind=P.POSITIONAL_OR_KEYWORD)
        pos_only_int = P("a", annotation=TypedValue(int), kind=P.POSITIONAL_ONLY)
        pos_kw_sig = Signature.make([pos_kw_int])
        self.can(pos_kw_sig, pos_kw_sig)
        self.can(pos_kw_sig, Signature.make([pos_kw_object]))
        self.cannot(pos_kw_sig, Signature.make([pos_kw_bool]))
        self.cannot(pos_kw_sig, Signature.make([pos_only_int]))
        self.cannot(pos_kw_sig, Signature.make([pos_kw_int_b]))
        self.cannot(
            pos_kw_sig,
            Signature.make([P("x", annotation=TupleInt, kind=P.VAR_POSITIONAL)]),
        )
        self.cannot(
            pos_kw_sig, Signature.make([P("x", annotation=DictInt, kind=P.VAR_KEYWORD)])
        )
        self.can(
            pos_kw_sig,
            Signature.make(
                [
                    P("x", annotation=TupleInt, kind=P.VAR_POSITIONAL),
                    P("y", annotation=DictInt, kind=P.VAR_KEYWORD),
                ]
            ),
        )
        self.can(
            pos_kw_sig,
            Signature.make(
                [
                    P("x", annotation=TupleObject, kind=P.VAR_POSITIONAL),
                    P("y", annotation=DictInt, kind=P.VAR_KEYWORD),
                ]
            ),
        )
        self.cannot(
            pos_kw_sig,
            Signature.make(
                [
                    P("x", annotation=TupleBool, kind=P.VAR_POSITIONAL),
                    P("y", annotation=DictInt, kind=P.VAR_KEYWORD),
                ]
            ),
        )

        # unhashable default
        seq_int = GenericValue(Sequence, [TypedValue(int)])
        list_default = P(
            "a",
            annotation=seq_int,
            default=KnownValue([]),
            kind=P.POSITIONAL_OR_KEYWORD,
        )
        sig = Signature.make([list_default])
        no_default_sig = Signature.make(
            [P("a", annotation=seq_int, kind=P.POSITIONAL_OR_KEYWORD)]
        )
        self.cannot(sig, no_default_sig)
        self.can(no_default_sig, sig)

    def test_kw_only(self) -> None:
        kw_only_int = P("a", annotation=TypedValue(int), kind=P.KEYWORD_ONLY)
        kw_only_int_b = P("b", annotation=TypedValue(int), kind=P.KEYWORD_ONLY)
        kw_only_bool = P("a", annotation=TypedValue(bool), kind=P.KEYWORD_ONLY)
        kw_only_object = P("a", annotation=TypedValue(object), kind=P.KEYWORD_ONLY)
        pos_only_int = P("a", annotation=TypedValue(int), kind=P.POSITIONAL_ONLY)
        pos_kw_int = P("a", annotation=TypedValue(int), kind=P.POSITIONAL_OR_KEYWORD)
        kw_only_sig = Signature.make([kw_only_int])
        self.can(kw_only_sig, kw_only_sig)
        self.can(kw_only_sig, Signature.make([kw_only_object]))
        self.cannot(kw_only_sig, Signature.make([kw_only_bool]))
        self.cannot(kw_only_sig, Signature.make([kw_only_int_b]))
        self.can(kw_only_sig, Signature.make([pos_kw_int]))
        self.cannot(kw_only_sig, Signature.make([pos_only_int]))
        self.can(
            kw_only_sig,
            Signature.make([P("x", annotation=DictInt, kind=P.VAR_KEYWORD)]),
        )
        self.cannot(
            kw_only_sig,
            Signature.make([P("x", annotation=DictBool, kind=P.VAR_KEYWORD)]),
        )

    def test_var_positional(self) -> None:
        var_pos_int = P("a", annotation=TupleInt, kind=P.VAR_POSITIONAL)
        var_pos_object = P("b", annotation=TupleObject, kind=P.VAR_POSITIONAL)
        var_pos_bool = P("c", annotation=TupleBool, kind=P.VAR_POSITIONAL)
        var_pos_sig = Signature.make([var_pos_int])
        self.can(var_pos_sig, var_pos_sig)
        self.can(var_pos_sig, Signature.make([var_pos_object]))
        self.cannot(var_pos_sig, Signature.make([var_pos_bool]))
        self.can(
            var_pos_sig,
            Signature.make(
                [
                    P("d", annotation=TypedValue(object), kind=P.POSITIONAL_ONLY),
                    var_pos_int,
                ]
            ),
        )
        self.cannot(
            var_pos_sig,
            Signature.make(
                [
                    P("d", annotation=TypedValue(bool), kind=P.POSITIONAL_ONLY),
                    var_pos_int,
                ]
            ),
        )

    def test_var_keyword(self) -> None:
        var_kw_int = P("a", annotation=DictInt, kind=P.VAR_KEYWORD)
        var_kw_object = P("b", annotation=DictObject, kind=P.VAR_KEYWORD)
        var_kw_bool = P("c", annotation=DictBool, kind=P.VAR_KEYWORD)
        var_kw_sig = Signature.make([var_kw_int])
        self.can(var_kw_sig, var_kw_sig)
        self.can(var_kw_sig, Signature.make([var_kw_object]))
        self.cannot(var_kw_sig, Signature.make([var_kw_bool]))
        self.can(
            var_kw_sig,
            Signature.make(
                [P("d", annotation=TypedValue(object), kind=P.KEYWORD_ONLY), var_kw_int]
            ),
        )
        self.cannot(
            var_kw_sig,
            Signature.make(
                [P("d", annotation=TypedValue(bool), kind=P.KEYWORD_ONLY), var_kw_int]
            ),
        )

    def test_advanced_var_positional(self) -> None:
        three_ints_sig = Signature.make(
            [
                P("a", annotation=TypedValue(int), kind=P.POSITIONAL_ONLY),
                P("b", annotation=TypedValue(int), kind=P.POSITIONAL_ONLY),
                P("c", annotation=TypedValue(int), kind=P.POSITIONAL_ONLY),
            ]
        )
        object_int = P(
            "args",
            annotation=SequenceIncompleteValue(
                tuple, [TypedValue(object), TypedValue(int)]
            ),
            kind=P.VAR_POSITIONAL,
        )
        self.can(
            three_ints_sig,
            Signature.make(
                [P("a", annotation=TypedValue(int), kind=P.POSITIONAL_ONLY), object_int]
            ),
        )
        self.cannot(three_ints_sig, Signature.make([object_int]))

    def test_advanced_var_keyword(self) -> None:
        three_ints_sig = Signature.make(
            [
                P("a", annotation=TypedValue(int), kind=P.KEYWORD_ONLY),
                P("b", annotation=TypedValue(int), kind=P.KEYWORD_ONLY),
                P("c", annotation=TypedValue(int), kind=P.KEYWORD_ONLY),
            ]
        )
        dict_int = P(
            "args",
            annotation=GenericValue(dict, [TypedValue(str), TypedValue(int)]),
            kind=P.VAR_KEYWORD,
        )
        self.can(three_ints_sig, Signature.make([dict_int]))
        good_td = TypedDictValue(
            {
                "a": (True, TypedValue(int)),
                "b": (True, TypedValue(int)),
                "c": (True, TypedValue(int)),
            }
        )
        self.can(
            three_ints_sig,
            Signature.make([P("a", annotation=good_td, kind=P.VAR_KEYWORD)]),
        )
        bad_td = TypedDictValue(
            {"a": (True, TypedValue(int)), "b": (True, TypedValue(int))}
        )
        self.cannot(
            three_ints_sig,
            Signature.make([P("a", annotation=bad_td, kind=P.VAR_KEYWORD)]),
        )


class TestProperty(TestNameCheckVisitorBase):
    @assert_passes()
    def test_property(self):
        from pyanalyze.tests import PropertyObject

        def capybara(uid):
            assert_is_value(PropertyObject(uid).string_property, TypedValue(str))


class TestShadowing(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def shadow_them(locals, __import__, *list, **dict):
            return (
                [int for int in list] * locals + __import__ + [v for v in dict.values()]
            )

        shadow_them(5, [1, 2], 3)


class TestCalls(TestNameCheckVisitorBase):
    @assert_fails(ErrorCode.incompatible_call)
    def test_too_few_args(self):
        def fn(x, y):
            return x + y

        def run():
            fn(1)

    @assert_passes()
    def test_correct_args(self):
        def fn(x, y):
            return x + y

        def run():
            fn(1, 2)

    @assert_fails(ErrorCode.incompatible_call)
    def test_wrong_kwarg(self):
        def fn(x, y=3):
            return x + y

        def run():
            fn(1, z=2)

    @assert_passes()
    def test_right_kwarg(self):
        def fn(x, y=3):
            return x + y

        def run():
            fn(1, y=2)

    @assert_passes()
    def test_classmethod_arg(self):
        class Capybara(object):
            @classmethod
            def hutia(cls):
                pass

            def tucotuco(self):
                self.hutia()

    @assert_passes()
    def test_staticmethod_arg(self):
        class Capybara(object):
            @staticmethod
            def hutia():
                pass

            def tucotuco(self):
                self.hutia()

    @assert_fails(ErrorCode.incompatible_call)
    def test_staticmethod_bad_arg(self):
        class Capybara(object):
            @staticmethod
            def hutia():
                pass

            def tucotuco(self):
                self.hutia(1)

    @assert_fails(ErrorCode.not_callable)
    def test_typ_call(self):
        def run(elts):
            lst = [x for x in elts]
            assert_is_value(lst, TypedValue(list))
            lst()

    @assert_passes()
    def test_override__call__(self):
        class WithCall(object):
            def __call__(self, arg):
                return arg * 2

        def capybara(x):
            obj = WithCall()
            assert_is_value(obj, TypedValue(WithCall))
            assert_is_value(obj(x), AnyValue(AnySource.from_another))

    @assert_fails(ErrorCode.incompatible_call)
    def test_unbound_method(self):
        class Capybara(object):
            def hutia(self, x=None):
                pass

            def tucotuco(self):
                self.hutia(y=2)

    @assert_fails(ErrorCode.undefined_attribute)
    def test_method_is_attribute(self):
        class Capybara(object):
            def __init__(self):
                self.tabs = self.tabs()

            def tabs(self):
                return []

            def hutia(self):
                self.tabs.append("hutia")

    @assert_passes()
    def test_type_inference_for_type_call(self):
        def fn():
            capybara = int("3")
            assert_is_value(capybara, TypedValue(int))

    @assert_passes()
    def test_return_value(self):
        from pyanalyze.value import HasAttrGuardExtension

        def capybara(x):
            l = hasattr(x, "foo")
            assert_is_value(
                l,
                AnnotatedValue(
                    TypedValue(bool),
                    [
                        HasAttrGuardExtension(
                            "object", KnownValue("foo"), AnyValue(AnySource.inference)
                        )
                    ],
                ),
            )

    @assert_passes()
    def test_required_kwonly_args(self):
        from pyanalyze.tests import takes_kwonly_argument

        def run():
            takes_kwonly_argument(1, kwonly_arg=True)

    @assert_fails(ErrorCode.incompatible_call)
    def test_missing_kwonly_arg(self):
        from pyanalyze.tests import takes_kwonly_argument

        def run():
            takes_kwonly_argument(1)

    @assert_fails(ErrorCode.incompatible_argument)
    def test_wrong_type_kwonly_arg(self):
        from pyanalyze.tests import takes_kwonly_argument

        def run():
            takes_kwonly_argument(1, kwonly_arg="capybara")

    @assert_fails(ErrorCode.incompatible_argument)
    def test_wrong_variable_name_value(self):
        def fn(qid):
            pass

        def capybara(uid):
            fn(uid)

    @assert_fails(ErrorCode.incompatible_argument)
    def test_wrong_variable_name_value_in_attr(self):
        def fn(qid):
            pass

        class Capybara(object):
            def __init__(self, uid):
                self.uid = uid

            def get_it(self):
                return fn(self.uid)

    @assert_fails(ErrorCode.incompatible_argument)
    def test_wrong_variable_name_value_in_subscript(self):
        def fn(qid):
            pass

        def render_item(self, item):
            return fn(item["uid"])

    @assert_passes()
    def test_kwargs(self):
        def fn(**kwargs):
            pass

        fn(uid=3)

    @assert_fails(ErrorCode.incompatible_argument)
    def test_known_argspec(self):
        def run():
            getattr(False, 42)

    @assert_fails(ErrorCode.incompatible_argument)
    def test_wrong_getattr_args(self):
        def run(attr):
            getattr(False, int(attr))

    @assert_passes()
    def test_kwonly_args(self):
        from pyanalyze.tests import KeywordOnlyArguments

        def capybara():
            return KeywordOnlyArguments(kwonly_arg="hydrochoerus")

    @assert_fails(ErrorCode.incompatible_call)
    def test_kwonly_args_bad_kwarg(self):
        from pyanalyze.tests import KeywordOnlyArguments

        class Capybara(KeywordOnlyArguments):
            def __init__(self):
                pass

        def run():
            Capybara(bad_kwarg="1")

    @assert_passes()
    def test_hasattr(self):
        class Quemisia(object):
            def gravis(self):
                if hasattr(self, "xaymaca"):
                    print(self.xaymaca)

    @assert_fails(ErrorCode.incompatible_call)
    def test_hasattr_wrong_args(self):
        def run():
            hasattr()

    @assert_fails(ErrorCode.incompatible_argument)
    def test_hasattr_mistyped_args(self):
        def run():
            hasattr(True, False)

    @assert_fails(ErrorCode.incompatible_call)
    def test_keyword_only_args(self):
        from pyanalyze.tests import KeywordOnlyArguments

        class Capybara(KeywordOnlyArguments):
            def __init__(self, neochoerus):
                pass

        def run():
            Capybara(hydrochoerus=None)

    @assert_passes()
    def test_correct_keyword_only_args(self):
        from pyanalyze.tests import KeywordOnlyArguments

        class Capybara(KeywordOnlyArguments):
            def __init__(self, neochoerus):
                pass

        def run():
            # This fails at runtime, but pyanalyze accepts it because of a special case
            # in pyanalyze.test_config.TestConfig.CLASS_TO_KEYWORD_ONLY_ARGUMENTS.
            Capybara(None, kwonly_arg="capybara")

    @assert_fails(ErrorCode.undefined_name)
    def test_undefined_args(self):
        def fn():
            return fn(*x)

    @assert_fails(ErrorCode.undefined_name)
    def test_undefined_kwargs(self):
        def fn():
            return fn(**x)

    @assert_passes()
    def test_set__name__(self):
        import pyanalyze.tests

        class A:
            def __init__(self) -> None:
                assert_is_value(self, TypedValue(A))

        A.__name__ = "B"
        A.__init__.__name__ = "B"

        def capybara():
            assert_is_value(A(), TypedValue(A))
            assert_is_value(
                pyanalyze.tests.WhatIsMyName(), TypedValue(pyanalyze.tests.WhatIsMyName)
            )


class TestTypeVar(TestNameCheckVisitorBase):
    @assert_passes()
    def test_simple(self):
        from typing import TypeVar, List, Generic

        T = TypeVar("T")

        def id(obj: T) -> T:
            return obj

        def get_one(obj: List[T]) -> T:
            for elt in obj:
                return elt
            assert False

        class GenCls(Generic[T]):
            def get_one(self: "GenCls[T]") -> T:
                raise NotImplementedError

            def get_another(self) -> T:
                raise NotImplementedError

        def capybara(x: str, xs: List[int], gen: GenCls[int]) -> None:
            assert_is_value(id(3), KnownValue(3))
            assert_is_value(id(x), TypedValue(str))
            assert_is_value(get_one(xs), TypedValue(int))
            assert_is_value(get_one([int(3)]), TypedValue(int))
            # This one doesn't work yet because we don't know how to go from
            # KnownValue([3]) to a GenericValue of some sort.
            # assert_is_value(get_one([3]), KnownValue(3))

            assert_is_value(gen.get_one(), TypedValue(int))
            assert_is_value(gen.get_another(), TypedValue(int))

    @assert_passes()
    def test_union_math(self):
        from typing import TypeVar, Optional

        T = TypeVar("T")

        def assert_not_none(arg: Optional[T]) -> T:
            assert arg is not None
            return arg

        def capybara(x: Optional[int]):
            assert_is_value(x, MultiValuedValue([KnownValue(None), TypedValue(int)]))
            assert_is_value(assert_not_none(x), TypedValue(int))

    @assert_passes()
    def test_only_T(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Capybara(Generic[T]):
            def add_one(self, obj: T) -> None:
                pass

        def capybara(x: Capybara[int]) -> None:
            x.add_one("x")  # E: incompatible_argument

    @assert_passes()
    def test_multi_typevar(self):
        from typing import TypeVar, Optional

        T = TypeVar("T")

        # inspired by tempfile.mktemp
        def mktemp(prefix: Optional[T] = None, suffix: Optional[T] = None) -> T:
            raise NotImplementedError

        def capybara() -> None:
            assert_is_value(mktemp(), AnyValue(AnySource.generic_argument))
            assert_is_value(mktemp(prefix="p"), KnownValue("p"))
            assert_is_value(mktemp(suffix="s"), KnownValue("s"))
            assert_is_value(mktemp("p", "s"), KnownValue("p") | KnownValue("s"))

    @assert_passes()
    def test_generic_base(self):
        from typing import TypeVar, Generic

        T = TypeVar("T")

        class Base(Generic[T]):
            pass

        class Derived(Base[int]):
            pass

        def take_base(b: Base[int]) -> None:
            pass

        def capybara(c: Derived):
            take_base(c)

    @assert_fails(ErrorCode.incompatible_argument)
    def test_wrong_generic_base(self):
        from typing import TypeVar, Generic

        T = TypeVar("T")

        class Base(Generic[T]):
            pass

        class Derived(Base[int]):
            pass

        def take_base(b: Base[str]) -> None:
            pass

        def capybara(c: Derived):
            take_base(c)

    @skip_before((3, 10))
    @assert_passes()
    def test_typeshed(self):
        from typing import List

        def capybara(lst: List[int]) -> None:
            lst.append("x")  # E: incompatible_argument

    @assert_passes()
    def test_generic_super(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class A(Generic[T]):
            def capybara(self) -> None:
                pass

        class B(A):
            def capybara(self) -> None:
                super().capybara()


class TestAllowCall(TestNameCheckVisitorBase):
    @assert_passes()
    def test_encode_decode(self):
        def capybara():
            s = "x"
            b = b"x"
            assert_is_value(s.encode("ascii"), KnownValue(b"x"))
            assert_is_value(b.decode("ascii"), KnownValue("x"))

            s.encode("not an encoding")  # E: incompatible_call


class TestAnnotated(TestNameCheckVisitorBase):
    @assert_passes()
    def test_preserve(self):
        from typing_extensions import Annotated
        from typing import TypeVar

        T = TypeVar("T")

        def f(x: T) -> T:
            return x

        def caller(x: Annotated[int, 42]):
            assert_is_value(x, AnnotatedValue(TypedValue(int), [KnownValue(42)]))
            assert_is_value(f(x), AnnotatedValue(TypedValue(int), [KnownValue(42)]))
