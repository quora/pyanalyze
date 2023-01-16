# static analysis: ignore
from collections.abc import Sequence

from .implementation import assert_is_value
from .signature import (
    ConcreteSignature,
    ELLIPSIS_PARAM,
    OverloadedSignature,
    ParameterKind as K,
    Signature,
    SigParameter as P,
)
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before
from .test_value import CTX
from .tests import make_simple_sequence

from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CanAssignError,
    GenericValue,
    KnownValue,
    SequenceValue,
    TypedDictValue,
    TypedValue,
)

TupleInt = GenericValue(tuple, [TypedValue(int)])
TupleBool = GenericValue(tuple, [TypedValue(bool)])
TupleObject = GenericValue(tuple, [TypedValue(object)])
DictInt = GenericValue(dict, [TypedValue(str), TypedValue(int)])
DictBool = GenericValue(dict, [TypedValue(str), TypedValue(bool)])
DictObject = GenericValue(dict, [TypedValue(str), TypedValue(object)])


def test_stringify() -> None:
    assert "() -> Any[unannotated]" == str(Signature.make([]))
    assert "() -> int" == str(Signature.make([], TypedValue(int)))
    assert "@asynq () -> int" == str(Signature.make([], TypedValue(int), is_asynq=True))
    assert "(...) -> int" == str(Signature.make([ELLIPSIS_PARAM], TypedValue(int)))
    assert "(x: int) -> int" == str(
        Signature.make([P("x", annotation=TypedValue(int))], TypedValue(int))
    )
    overload = OverloadedSignature(
        [
            Signature.make([], TypedValue(str)),
            Signature.make([P("x", annotation=TypedValue(int))], TypedValue(int)),
        ]
    )
    assert str(overload) == "overloaded (() -> str, (x: int) -> int)"


class TestCanAssign:
    def can(self, left: ConcreteSignature, right: ConcreteSignature) -> None:
        tv_map = left.can_assign(right, CTX)
        assert isinstance(
            tv_map, dict
        ), f"cannot assign {right} to {left} due to {tv_map}"

    def cannot(self, left: ConcreteSignature, right: ConcreteSignature) -> None:
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
        pos_only_int = P("x", annotation=TypedValue(int), kind=K.POSITIONAL_ONLY)
        pos_only_object = P("y", annotation=TypedValue(object), kind=K.POSITIONAL_ONLY)
        pos_only_bool = P("z", annotation=TypedValue(bool), kind=K.POSITIONAL_ONLY)
        pos_kw_int = P("a", annotation=TypedValue(int), kind=K.POSITIONAL_OR_KEYWORD)
        pos_only_with_default = P(
            "d",
            annotation=TypedValue(int),
            kind=K.POSITIONAL_ONLY,
            default=KnownValue(0),
        )
        pos_only_sig = Signature.make([pos_only_int])
        self.can(pos_only_sig, pos_only_sig)
        self.can(pos_only_sig, Signature.make([pos_kw_int]))
        self.can(pos_only_sig, Signature.make([pos_only_object]))
        self.can(pos_only_sig, Signature.make([pos_only_with_default]))
        self.can(pos_only_sig, Signature.make([pos_only_int, pos_only_with_default]))
        self.cannot(Signature.make([pos_only_with_default]), pos_only_sig)
        self.cannot(pos_only_sig, Signature.make([pos_only_int, pos_only_bool]))
        self.cannot(pos_only_sig, Signature.make([pos_only_bool]))
        self.cannot(
            pos_only_sig,
            Signature.make([P("x", annotation=TypedValue(int), kind=K.KEYWORD_ONLY)]),
        )

        # *args interaction
        self.can(
            pos_only_sig,
            Signature.make([P("whatever", annotation=TupleInt, kind=K.VAR_POSITIONAL)]),
        )
        self.cannot(
            pos_only_sig,
            Signature.make([P("x", annotation=TupleBool, kind=K.VAR_POSITIONAL)]),
        )

    def test_pos_or_keyword(self) -> None:
        pos_kw_int = P("a", annotation=TypedValue(int), kind=K.POSITIONAL_OR_KEYWORD)
        pos_kw_int_b = P("b", annotation=TypedValue(int), kind=K.POSITIONAL_OR_KEYWORD)
        pos_kw_object = P(
            "a", annotation=TypedValue(object), kind=K.POSITIONAL_OR_KEYWORD
        )
        pos_kw_bool = P("a", annotation=TypedValue(bool), kind=K.POSITIONAL_OR_KEYWORD)
        pos_only_int = P("a", annotation=TypedValue(int), kind=K.POSITIONAL_ONLY)
        pos_kw_with_default = P(
            "d",
            annotation=TypedValue(int),
            kind=K.POSITIONAL_OR_KEYWORD,
            default=KnownValue(0),
        )
        pos_kw_sig = Signature.make([pos_kw_int])
        self.can(pos_kw_sig, pos_kw_sig)
        self.can(pos_kw_sig, Signature.make([pos_kw_object]))
        self.can(pos_kw_sig, Signature.make([pos_kw_int, pos_kw_with_default]))
        self.cannot(pos_kw_sig, Signature.make([pos_kw_int, pos_kw_int_b]))
        self.cannot(pos_kw_sig, Signature.make([pos_kw_bool]))
        self.cannot(pos_kw_sig, Signature.make([pos_only_int]))
        self.cannot(pos_kw_sig, Signature.make([pos_kw_int_b]))
        self.cannot(
            pos_kw_sig,
            Signature.make([P("x", annotation=TupleInt, kind=K.VAR_POSITIONAL)]),
        )
        self.cannot(
            pos_kw_sig, Signature.make([P("x", annotation=DictInt, kind=K.VAR_KEYWORD)])
        )
        self.can(
            pos_kw_sig,
            Signature.make(
                [
                    P("x", annotation=TupleInt, kind=K.VAR_POSITIONAL),
                    P("y", annotation=DictInt, kind=K.VAR_KEYWORD),
                ]
            ),
        )
        self.can(
            pos_kw_sig,
            Signature.make(
                [
                    P("x", annotation=TupleObject, kind=K.VAR_POSITIONAL),
                    P("y", annotation=DictInt, kind=K.VAR_KEYWORD),
                ]
            ),
        )
        self.cannot(
            pos_kw_sig,
            Signature.make(
                [
                    P("x", annotation=TupleBool, kind=K.VAR_POSITIONAL),
                    P("y", annotation=DictInt, kind=K.VAR_KEYWORD),
                ]
            ),
        )

        # unhashable default
        seq_int = GenericValue(Sequence, [TypedValue(int)])
        list_default = P(
            "a",
            annotation=seq_int,
            default=KnownValue([]),
            kind=K.POSITIONAL_OR_KEYWORD,
        )
        sig = Signature.make([list_default])
        no_default_sig = Signature.make(
            [P("a", annotation=seq_int, kind=K.POSITIONAL_OR_KEYWORD)]
        )
        self.cannot(sig, no_default_sig)
        self.can(no_default_sig, sig)

    def test_kw_only(self) -> None:
        kw_only_int = P("a", annotation=TypedValue(int), kind=K.KEYWORD_ONLY)
        kw_only_int_b = P("b", annotation=TypedValue(int), kind=K.KEYWORD_ONLY)
        kw_only_bool = P("a", annotation=TypedValue(bool), kind=K.KEYWORD_ONLY)
        kw_only_object = P("a", annotation=TypedValue(object), kind=K.KEYWORD_ONLY)
        pos_only_int = P("a", annotation=TypedValue(int), kind=K.POSITIONAL_ONLY)
        pos_kw_int = P("a", annotation=TypedValue(int), kind=K.POSITIONAL_OR_KEYWORD)
        kw_only_sig = Signature.make([kw_only_int])
        self.can(kw_only_sig, kw_only_sig)
        self.can(kw_only_sig, Signature.make([kw_only_object]))
        self.cannot(kw_only_sig, Signature.make([kw_only_bool]))
        self.cannot(kw_only_sig, Signature.make([kw_only_int_b]))
        self.can(kw_only_sig, Signature.make([pos_kw_int]))
        self.cannot(kw_only_sig, Signature.make([pos_only_int]))
        self.can(
            kw_only_sig,
            Signature.make([P("x", annotation=DictInt, kind=K.VAR_KEYWORD)]),
        )
        self.cannot(
            kw_only_sig,
            Signature.make([P("x", annotation=DictBool, kind=K.VAR_KEYWORD)]),
        )
        self.cannot(Signature.make([]), kw_only_sig)

    def test_var_positional(self) -> None:
        var_pos_int = P("a", annotation=TupleInt, kind=K.VAR_POSITIONAL)
        var_pos_object = P("b", annotation=TupleObject, kind=K.VAR_POSITIONAL)
        var_pos_bool = P("c", annotation=TupleBool, kind=K.VAR_POSITIONAL)
        var_pos_sig = Signature.make([var_pos_int])
        self.can(var_pos_sig, var_pos_sig)
        self.can(var_pos_sig, Signature.make([var_pos_object]))
        self.cannot(var_pos_sig, Signature.make([var_pos_bool]))
        self.can(
            var_pos_sig,
            Signature.make(
                [
                    P(
                        "d",
                        annotation=TypedValue(object),
                        kind=K.POSITIONAL_ONLY,
                        default=KnownValue(True),
                    ),
                    var_pos_int,
                ]
            ),
        )
        self.cannot(
            var_pos_sig,
            Signature.make(
                [
                    P(
                        "d",
                        annotation=TypedValue(bool),
                        kind=K.POSITIONAL_ONLY,
                        default=KnownValue(True),
                    ),
                    var_pos_int,
                ]
            ),
        )

    def test_var_keyword(self) -> None:
        var_kw_int = P("a", annotation=DictInt, kind=K.VAR_KEYWORD)
        var_kw_object = P("b", annotation=DictObject, kind=K.VAR_KEYWORD)
        var_kw_bool = P("c", annotation=DictBool, kind=K.VAR_KEYWORD)
        var_kw_sig = Signature.make([var_kw_int])
        self.can(var_kw_sig, var_kw_sig)
        self.can(var_kw_sig, Signature.make([var_kw_object]))
        self.cannot(var_kw_sig, Signature.make([var_kw_bool]))
        self.can(
            var_kw_sig,
            Signature.make(
                [
                    P(
                        "d",
                        annotation=TypedValue(object),
                        kind=K.KEYWORD_ONLY,
                        default=KnownValue(True),
                    ),
                    var_kw_int,
                ]
            ),
        )
        self.cannot(
            var_kw_sig,
            Signature.make(
                [
                    P(
                        "d",
                        annotation=TypedValue(bool),
                        kind=K.KEYWORD_ONLY,
                        default=KnownValue(True),
                    ),
                    var_kw_int,
                ]
            ),
        )

    def test_advanced_var_positional(self) -> None:
        three_ints_sig = Signature.make(
            [
                P("a", annotation=TypedValue(int), kind=K.POSITIONAL_ONLY),
                P("b", annotation=TypedValue(int), kind=K.POSITIONAL_ONLY),
                P("c", annotation=TypedValue(int), kind=K.POSITIONAL_ONLY),
            ]
        )
        object_int = P(
            "args",
            annotation=make_simple_sequence(
                tuple, [TypedValue(object), TypedValue(int)]
            ),
            kind=K.VAR_POSITIONAL,
        )
        self.can(
            three_ints_sig,
            Signature.make(
                [P("a", annotation=TypedValue(int), kind=K.POSITIONAL_ONLY), object_int]
            ),
        )
        self.cannot(three_ints_sig, Signature.make([object_int]))

    def test_advanced_var_keyword(self) -> None:
        three_ints_sig = Signature.make(
            [
                P("a", annotation=TypedValue(int), kind=K.KEYWORD_ONLY),
                P("b", annotation=TypedValue(int), kind=K.KEYWORD_ONLY),
                P("c", annotation=TypedValue(int), kind=K.KEYWORD_ONLY),
            ]
        )
        dict_int = P(
            "args",
            annotation=GenericValue(dict, [TypedValue(str), TypedValue(int)]),
            kind=K.VAR_KEYWORD,
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
            Signature.make([P("a", annotation=good_td, kind=K.VAR_KEYWORD)]),
        )
        smaller_td = TypedDictValue(
            {"a": (True, TypedValue(int)), "b": (True, TypedValue(int))}
        )
        # This is still OK because TypedDicts are allowed to have extra keys.
        # TODO change to can
        self.cannot(
            three_ints_sig,
            Signature.make([P("a", annotation=smaller_td, kind=K.VAR_KEYWORD)]),
        )
        bad_td = TypedDictValue(
            {"a": (True, TypedValue(str)), "b": (True, TypedValue(int))}
        )
        self.cannot(
            three_ints_sig,
            Signature.make([P("a", annotation=bad_td, kind=K.VAR_KEYWORD)]),
        )

    def test_overloads(self) -> None:
        sig1 = Signature.make([], TypedValue(str))
        sig2 = Signature.make([P("x", annotation=TypedValue(int))], TypedValue(int))
        overload = OverloadedSignature([sig1, sig2])
        self.can(overload, overload)
        self.can(sig1, overload)
        self.can(sig2, overload)
        self.cannot(overload, sig1)
        self.cannot(overload, sig2)

        sig3 = Signature.make([P("y", annotation=TypedValue(float))], TypedValue(float))
        overload2 = OverloadedSignature([sig1, sig2, sig3])
        self.can(overload, overload2)
        self.cannot(overload2, overload)
        self.can(sig1, overload2)
        self.cannot(sig3, overload)

    def test_asynq(self) -> None:
        ret = AnyValue(AnySource.unannotated)
        self.can(Signature.make([], ret), Signature.make([], ret, is_asynq=True))
        self.cannot(Signature.make([], ret, is_asynq=True), Signature.make([], ret))
        self.can(
            Signature.make([], ret, is_asynq=True),
            Signature.make([P("", K.ELLIPSIS)], ret),
        )
        self.can(
            Signature.make(
                [P("x", K.POSITIONAL_ONLY, annotation=TypedValue(int))],
                ret,
                is_asynq=True,
            ),
            Signature.make([P("", K.ELLIPSIS)], ret),
        )
        self.cannot(
            Signature.make(
                [P("x", K.POSITIONAL_ONLY, annotation=TypedValue(int))],
                ret,
                is_asynq=True,
            ),
            Signature.make(
                [
                    P("x", K.POSITIONAL_ONLY, annotation=TypedValue(str)),
                    P("", K.ELLIPSIS),
                ],
                ret,
            ),
        )


class TestProperty(TestNameCheckVisitorBase):
    @assert_passes()
    def test_property(self):
        from pyanalyze.tests import PropertyObject

        def capybara(uid):
            assert_is_value(PropertyObject(uid).string_property, TypedValue(str))

    @assert_passes()
    def test_local_return(self):
        class X:
            @property
            def foo(self):
                return str(1)

        def capybara() -> None:
            assert_is_value(X().foo, TypedValue(str))


class TestShadowing(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def shadow_them(locals, __import__, *list, **dict):
            return (
                [int for int in list] * locals + __import__ + [v for v in dict.values()]
            )

        shadow_them(5, [1, 2], 3)


class TestCalls(TestNameCheckVisitorBase):
    @assert_passes()
    def test_error_location(self):
        def two_args(x: str, y: int) -> None:
            pass

        def capybara():
            two_args(
                (
                    "one very long string so long that it goes on its own line is this"
                    " enough"
                ),
                "not an int",  # E: incompatible_argument
            )

    @assert_passes()
    def test_too_few_args(self):
        def fn(x, y):
            return x + y

        def run():
            fn(1)  # E: incompatible_call

    @assert_passes()
    def test_correct_args(self):
        def fn(x, y):
            return x + y

        def run():
            fn(1, 2)

    @assert_passes()
    def test_wrong_kwarg(self):
        def fn(x, y=3):
            return x + y

        def run():
            fn(1, z=2)  # E: incompatible_call

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

    @assert_passes()
    def test_staticmethod_bad_arg(self):
        class Capybara(object):
            @staticmethod
            def hutia():
                pass

            def tucotuco(self):
                self.hutia(1)  # E: incompatible_call

    @assert_passes()
    def test_typ_call(self):
        def run(elts):
            lst = [x for x in elts]
            assert_is_value(
                lst, SequenceValue(list, [(True, AnyValue(AnySource.generic_argument))])
            )
            lst()  # E: not_callable

    @assert_passes()
    def test_override__call__(self):
        class WithCall(object):
            def __call__(self, arg):
                return arg * 2

        def capybara(x):
            obj = WithCall()
            assert_is_value(obj, TypedValue(WithCall))
            assert_is_value(obj(x), AnyValue(AnySource.unannotated))

    @assert_passes()
    def test_unbound_method(self):
        class Capybara(object):
            def hutia(self, x=None):
                pass

            def tucotuco(self):
                self.hutia(y=2)  # E: incompatible_call

    @assert_passes()
    def test_method_is_attribute(self):
        class Capybara(object):
            def __init__(self):
                self.tabs = self.tabs()

            def tabs(self):
                return []

            def hutia(self):
                self.tabs.append("hutia")  # E: undefined_attribute

    @assert_passes()
    def test_type_inference_for_type_call(self):
        def fn():
            capybara = int("3")
            assert_is_value(capybara, TypedValue(int))

    @assert_passes()
    def test_required_kwonly_args(self):
        from pyanalyze.tests import takes_kwonly_argument

        def run():
            takes_kwonly_argument(1, kwonly_arg=True)

    @assert_passes()
    def test_missing_kwonly_arg(self):
        from pyanalyze.tests import takes_kwonly_argument

        def run():
            takes_kwonly_argument(1)  # E: incompatible_call

    @assert_passes()
    def test_wrong_type_kwonly_arg(self):
        from pyanalyze.tests import takes_kwonly_argument

        def run():
            takes_kwonly_argument(1, kwonly_arg="capybara")  # E: incompatible_argument

    @assert_passes()
    def test_wrong_variable_name_value(self):
        def fn(qid):
            pass

        def capybara(uid):
            fn(uid)  # E: incompatible_argument

    @assert_passes()
    def test_wrong_variable_name_value_in_attr(self):
        def fn(qid):
            pass

        class Capybara(object):
            def __init__(self, uid):
                self.uid = uid

            def get_it(self):
                return fn(self.uid)  # E: incompatible_argument

    @assert_passes()
    def test_wrong_variable_name_value_in_subscript(self):
        def fn(qid):
            pass

        def render_item(self, item):
            return fn(item["uid"])  # E: incompatible_argument

    @assert_passes()
    def test_kwargs(self):
        def fn(**kwargs):
            pass

        fn(uid=3)

    @assert_passes()
    def test_known_argspec(self):
        def run():
            getattr(False, 42)  # E: incompatible_argument

    @assert_passes()
    def test_wrong_getattr_args(self):
        def run(attr):
            getattr(False, int(attr))  # E: incompatible_argument

    @assert_passes()
    def test_kwonly_args(self):
        from pyanalyze.tests import KeywordOnlyArguments

        def capybara():
            return KeywordOnlyArguments(kwonly_arg="hydrochoerus")

    @assert_passes()
    def test_kwonly_args_bad_kwarg(self):
        from pyanalyze.tests import KeywordOnlyArguments

        class Capybara(KeywordOnlyArguments):
            def __init__(self):
                pass

        def run():
            Capybara(bad_kwarg="1")  # E: incompatible_call

    @assert_passes()
    def test_hasattr(self):
        from dataclasses import dataclass
        from typing import Any, cast

        from qcore.testing import Anything

        @dataclass
        class Quemisia(object):
            def gravis(self):
                if hasattr(self, "xaymaca"):
                    print(self.xaymaca)

        def now_it_has_it(o: object):
            o.xaymaca  # E: undefined_attribute
            if hasattr(o, "xaymaca"):
                print(o.xaymaca)

        def wrong_args():
            hasattr()  # E: incompatible_call

        def mistyped_args():
            hasattr(True, False)  # E: incompatible_argument

        # The HasAttrGuard gets stripped.
        inferred = AnnotatedValue(TypedValue(bool), [cast(Any, Anything)])

        def only_on_class(o: object):
            val = hasattr(o, "__qualname__")
            assert_is_value(val, inferred)

    @assert_passes()
    def test_keyword_only_args(self):
        from pyanalyze.tests import KeywordOnlyArguments

        class Capybara(KeywordOnlyArguments):
            def __init__(self, neochoerus):
                pass

        def run():
            Capybara(hydrochoerus=None)  # E: incompatible_call

    @assert_passes()
    def test_correct_keyword_only_args(self):
        from pyanalyze.tests import KeywordOnlyArguments

        class Capybara(KeywordOnlyArguments):
            def __init__(self, neochoerus):
                pass

        def run():
            # This fails at runtime, but pyanalyze accepts it because of a special case
            # in pyanalyze.test_config.get_constructor.
            Capybara(None, kwonly_arg="capybara")

    @assert_passes()
    def test_undefined_args(self):
        def fn(*args):
            return fn(*x)  # E: undefined_name

    @assert_passes()
    def test_undefined_kwargs(self):
        def fn(**kwargs):
            return fn(**x)  # E: undefined_name

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

    @assert_passes()
    def test_star_args(self):
        from typing import Dict, Sequence

        def takes_args(*args: int) -> None:
            pass

        def takes_a_b(a: int, b: int) -> None:
            pass

        def capybara(
            ints: Sequence[int], strs: Sequence[str], d: Dict[str, int]
        ) -> None:
            takes_args(*1)  # E: incompatible_call
            takes_args(*("a",))  # E: incompatible_argument
            takes_args(*(1, 2))  # ok
            takes_args(1, 2)  # ok
            takes_args("a")  # E: incompatible_argument
            takes_args()  # ok
            takes_args(*ints)  # ok
            takes_args(*strs)  # E: incompatible_argument
            takes_args(args=3)  # E: incompatible_call

            takes_a_b(1, 2)  # ok
            takes_a_b(*(1, 2))  # ok
            takes_a_b(*(1, 2, 3))  # E: incompatible_call
            takes_a_b(1, 2, *ints)  # E: incompatible_call
            takes_a_b(*ints)  # ok
            takes_a_b(1, *ints)  # ok
            takes_a_b(1)  # E: incompatible_call
            takes_a_b()  # E: incompatible_call
            takes_a_b(1, 2, **d)  # E: incompatible_call
            takes_a_b(**d)  # ok

    @assert_passes()
    def test_star_kwargs(self):
        from typing import Any, Dict, Sequence

        from typing_extensions import NotRequired, TypedDict

        def takes_ab(a: int, b: str = "") -> None:
            pass

        class TD1(TypedDict):
            a: int
            b: NotRequired[str]

        class TD2(TypedDict):
            a: NotRequired[int]
            b: str

        def capybara(
            good_dict: Dict[str, Any],
            bad_dict: Dict[str, int],
            very_bad_dict: Dict[int, str],
            any_dict: Dict[Any, Any],
            anys: Sequence[Any],
            td1: TD1,
            td2: TD2,
            cond: Any,
        ) -> None:
            takes_ab(**{"a": 1})  # ok
            takes_ab(**{"b": ""})  # E: incompatible_call
            takes_ab(**good_dict)  # ok
            takes_ab(**bad_dict)  # E: incompatible_argument
            takes_ab(**("not", "a", "dict"))  # E: incompatible_call
            takes_ab(**very_bad_dict)  # E: incompatible_call
            takes_ab(**any_dict)  # ok
            takes_ab(**{1: ""})  # E: incompatible_call
            takes_ab(**td1)  # ok
            takes_ab(**td2)  # E: incompatible_call
            td1_or_2 = td1 if cond else td2
            takes_ab(**td1_or_2)  # E: incompatible_call
            td1_or_good = td1 if cond else good_dict
            takes_ab(**td1_or_good)  # ok
            takes_ab(**{"a": 1})  # ok
            takes_ab(**{"a": cond})  # ok
            takes_ab(**{cond: cond})  # ok
            takes_ab(**{"a": 1, cond: ""})  # ok
            takes_ab(**{"b": ""})  # E: incompatible_call
            bad_div_or_good = {"b": ""} if cond else good_dict
            # TODO ideally this should error, because if we take the {"b": ""}
            # branch we're missing a required arg.
            takes_ab(**bad_div_or_good)

            takes_ab(a=3, **td1)  # E: incompatible_call
            takes_ab(**{"a": 3}, **td1)  # E: incompatible_call
            takes_ab(**{"a": 3}, **{"b": ""})  # ok

            takes_ab(1, a=1)  # E: incompatible_call
            takes_ab(*anys, a=1)  # E: incompatible_call

    @assert_passes()
    def test_union_key(self):
        def many_args(a: int = 0, b: int = 1, c: int = 2) -> None:
            pass

        def capybara(arg):
            kwargs = {key: 1 for key in ("a", "b", "c")}
            many_args(**kwargs)  # ok
            bad_kwargs = {key: 1 for key in ("a", "b", "c", "d")}
            many_args(**bad_kwargs)  # E: incompatible_call

            div = {}
            for key in ("a", "b", "c"):
                div[key] = 1
            many_args(**div)  # ok
            div["d"] = 3
            many_args(**div)  # E: incompatible_call

            known_int_kwargs = {i: i for i in (1, 2, 3)}
            many_args(**known_int_kwargs)  # E: incompatible_call
            typed_int_kwargs = {int(x): 1 for x in arg}
            many_args(**typed_int_kwargs)  # E: incompatible_call

    @skip_before((3, 8))
    def test_pos_only(self):
        self.assert_passes(
            """
            from typing import Sequence

            def pos_only(pos: int, /) -> None:
                pass

            def capybara(ints: Sequence[int], strs: Sequence[str]) -> None:
                pos_only(1)
                pos_only(*ints)
                pos_only(*strs)  # E: incompatible_argument
                pos_only(pos=1)  # E: incompatible_call
                pos_only()  # E: incompatible_call
                pos_only(1, 2)  # E: incompatible_call
            """
        )

    @assert_passes()
    def test_kw_only(self):
        from typing_extensions import NotRequired, TypedDict

        class TD(TypedDict):
            a: int
            b: NotRequired[str]

        class BadTD(TypedDict):
            a: NotRequired[int]
            b: NotRequired[str]

        def kwonly(*, a: int, b: str = "") -> None:
            pass

        def capybara(td: TD, bad_td: BadTD) -> None:
            kwonly(1)  # E: incompatible_call
            kwonly()  # E: incompatible_call
            kwonly(a=1)  # ok
            kwonly(b="")  # E: incompatible_call
            kwonly(a=1, b="")  # ok
            kwonly(**td)  # ok
            kwonly(**bad_td)  # E: incompatible_call


class TestAllowCall(TestNameCheckVisitorBase):
    @assert_passes()
    def test_encode_decode(self):
        def capybara():
            s = "x"
            b = b"x"
            assert_is_value(s.encode("ascii"), KnownValue(b"x"))
            assert_is_value(b.decode("ascii"), KnownValue("x"))

            s.encode("not an encoding")  # E: incompatible_call

    @assert_passes()
    def test_annotated_known(self):
        from typing_extensions import Annotated, Literal

        from pyanalyze.extensions import LiteralOnly

        def capybara():
            encoding: Annotated[Literal["ascii"], LiteralOnly()] = "ascii"

            s = "x"
            assert_is_value(s.encode(encoding), KnownValue(b"x"))


class TestOverload(TestNameCheckVisitorBase):
    @assert_passes()
    def test_overloaded_impl(self):
        from pyanalyze.tests import overloaded

        def capybara():
            assert_is_value(overloaded(), TypedValue(int))
            assert_is_value(overloaded("x"), TypedValue(str))
            overloaded(1)  # E: incompatible_argument
            overloaded("x", "y")  # E: incompatible_call

    @assert_passes()
    def test_pyanalyze_extensions(self):
        from typing import Union

        from pyanalyze.extensions import overload

        @overload
        def overloaded() -> int:
            raise NotImplementedError

        @overload
        def overloaded(x: str) -> str:
            raise NotImplementedError

        def overloaded(*args: str) -> Union[int, str]:
            if not args:
                return 0
            elif len(args) == 1:
                return args[0]
            else:
                raise TypeError("too many arguments")

        def capybara():
            assert_is_value(overloaded(), TypedValue(int))
            assert_is_value(overloaded("x"), TypedValue(str))
            overloaded(1)  # E: incompatible_argument
            overloaded("a", "b")  # E: incompatible_call

    @assert_passes()
    def test_typing_extensions(self):
        from typing import Union

        from typing_extensions import overload

        @overload
        def overloaded() -> int:
            raise NotImplementedError

        @overload
        def overloaded(x: str) -> str:
            raise NotImplementedError

        def overloaded(*args: str) -> Union[int, str]:
            if not args:
                return 0
            elif len(args) == 1:
                return args[0]
            else:
                raise TypeError("too many arguments")

        def capybara():
            assert_is_value(overloaded(), TypedValue(int))
            assert_is_value(overloaded("x"), TypedValue(str))
            overloaded(1)  # E: incompatible_argument
            overloaded("a", "b")  # E: incompatible_call

    @assert_passes()
    def test_list_any(self):
        from typing import Any, List, Union

        from typing_extensions import Literal

        from pyanalyze.extensions import overload

        @overload
        def overloaded(lst: List[str]) -> Literal[2]:
            pass

        @overload
        def overloaded(lst: List[int]) -> Literal[3]:
            pass

        def overloaded(lst: List[Any]) -> int:
            raise NotImplementedError

        def capybara(
            unannotated,
            list_union: List[Union[int, str]],
            union_list: Union[List[int], List[str]],
            explicit: Any,
        ):
            assert_is_value(overloaded(["x"]), KnownValue(2))
            assert_is_value(overloaded([1]), KnownValue(3))
            val = overloaded([])  # pyright and mypy: Literal[2]
            assert_is_value(val, AnyValue(AnySource.multiple_overload_matches))
            val2 = overloaded([unannotated])  # pyright: Literal[2], mypy: Any
            assert_is_value(val2, AnyValue(AnySource.multiple_overload_matches))
            val3 = overloaded(unannotated)  # pyright: Literal[2], mypy: Any
            assert_is_value(val3, AnyValue(AnySource.multiple_overload_matches))
            # pyright: Unknown, mypy: Literal[2]
            val4 = overloaded(list_union)  # E: incompatible_argument
            assert_is_value(val4, AnyValue(AnySource.error))
            val5 = overloaded(union_list)  # pyright and mypy: Literal[2, 3]
            assert_is_value(val5, KnownValue(3) | KnownValue(2))
            val6 = overloaded(explicit)  # pyright: Literal[2], mypy: Any
            assert_is_value(val6, AnyValue(AnySource.multiple_overload_matches))

    @assert_passes()
    def test_any_and_union(self):
        from typing import Any, Union

        from typing_extensions import Literal

        from pyanalyze.extensions import overload

        @overload
        def overloaded1(x: Any, y: str) -> Literal[2]:
            pass

        @overload
        def overloaded1(x: str, y: int) -> Literal[3]:
            pass

        def overloaded1(x: object, y: object) -> Literal[2, 3]:
            raise NotImplementedError

        @overload
        def overloaded2(x: object, y: str) -> Literal[2]:
            pass

        @overload
        def overloaded2(x: str, y: int) -> Literal[3]:
            pass

        def overloaded2(x: object, y: object) -> Literal[2, 3]:
            raise NotImplementedError

        def capybara(
            unannotated,
            int_or_str: Union[int, str],
            int_or_str_or_float: Union[int, str, float],
        ):
            val1 = overloaded1(unannotated, int_or_str)
            assert_is_value(val1, AnyValue(AnySource.multiple_overload_matches))
            val2 = overloaded2(unannotated, int_or_str)
            assert_is_value(val2, AnyValue(AnySource.multiple_overload_matches))
            val3 = overloaded2("x", int_or_str_or_float)  # E: incompatible_argument
            assert_is_value(val3, AnyValue(AnySource.error))

    @assert_passes()
    def test_same_return(self):
        from typing import Any

        from typing_extensions import Literal

        from pyanalyze.extensions import overload

        @overload
        def overloaded1(x: Any, y: str) -> Literal[2]:
            pass

        @overload
        def overloaded1(x: str, y: int) -> Literal[2]:
            pass

        def overloaded1(x: object, y: object) -> Literal[2]:
            raise NotImplementedError

        def capybara(x, y):
            # If multiple overloads match but have the same return type,
            # don't fall back to Any. This comes up in practice with str.__new__
            # in typeshed.
            assert_is_value(overloaded1(x, y), KnownValue(2))

    @assert_passes()
    def test_nested_class(self):
        class Outer:
            class Inner:
                pass

        def capybara():
            assert_is_value(Outer.Inner(), TypedValue(Outer.Inner))

    @assert_passes()
    def test_typeshed_overload(self):
        class SupportsWrite:
            def write(self, s: str) -> None:
                pass

        class SupportsWriteAndFlush(SupportsWrite):
            def flush(self) -> None:
                pass

        def capybara():
            print()  # ok
            print("x", file=SupportsWrite())
            print("x", file=SupportsWrite(), flush=True)  # E: incompatible_argument
            print("x", file=SupportsWriteAndFlush(), flush=True)
            print("x", file=SupportsWriteAndFlush())
            print("x", file="not a file")  # E: incompatible_argument

        def pacarana(f: float):
            assert_is_value(f.__round__(), TypedValue(int))
            assert_is_value(f.__round__(None), TypedValue(int))
            f.__round__(ndigits=None)  # E: incompatible_call
            assert_is_value(f.__round__(1), TypedValue(float))

    @assert_passes()
    def test_runtime(self):
        from pyanalyze.extensions import (
            overload as extension_overload,
            patch_typing_overload,
        )

        patch_typing_overload()
        from typing import overload

        assert overload is extension_overload, (overload, extension_overload)

        @overload
        def f(x: int) -> str:
            pass

        @overload
        def f(x: str) -> int:
            pass

        def f(x: object) -> object:
            raise NotImplementedError

        def capybara():
            assert_is_value(f(1), TypedValue(str))
            assert_is_value(f(""), TypedValue(int))

    @assert_passes()
    def test_bound_args_first(self):
        from pyanalyze.extensions import overload

        @overload
        def f(x: int, y: str) -> None:
            pass

        @overload
        def f(x: int) -> int:
            pass

        @overload
        def f(x: str) -> str:
            pass

        def f(x: object, y: object = ...) -> object:
            raise NotImplementedError

        def capybara():
            f(1.0)  # E: incompatible_argument
            f(1, 1)  # E: incompatible_argument
            f(1, 1, 1)  # E: incompatible_call

    @assert_passes()
    def test_ellipsis_default(self):
        from pyanalyze.extensions import assert_type

        def wrapper():
            from _pyanalyze_tests.overloaded import func

            assert_type(func(1), int)
            assert_type(func(1, 1), int)
            assert_type(func("x"), float)

    @assert_passes()
    def test_overloaded_method(self):
        from pyanalyze.extensions import assert_type, overload

        class Capybara:
            @overload
            def meth(self, x: int) -> str:
                ...

            @overload
            def meth(self, x: str) -> int:
                ...

            def meth(self, x: object) -> object:
                raise NotImplementedError

        cap = Capybara()

        def caller(inst: Capybara) -> None:
            assert_type(cap.meth(1), str)
            assert_type(cap.meth(""), int)

            assert_type(inst.meth(1), str)
            assert_type(inst.meth(""), int)


class TestSelfAnnotation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_method(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Capybara(Generic[T]):
            def method(self: "Capybara[int]") -> int:
                return 1

        def caller(ci: Capybara[int], cs: Capybara[str]):
            assert_is_value(ci.method(), TypedValue(int))
            cs.method()  # E: incompatible_argument

    @assert_passes()
    def test_property(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Capybara(Generic[T]):
            @property
            def prop(self: "Capybara[int]") -> int:
                return 1

        def caller(ci: Capybara[int], cs: Capybara[str]):
            assert_is_value(ci.prop, TypedValue(int))
            cs.prop  # E: incompatible_argument


class TestUnpack(TestNameCheckVisitorBase):
    @assert_passes()
    def test_args(self):
        from typing import Tuple

        from typing_extensions import Unpack

        def f(*args: Unpack[Tuple[int, str]]) -> None:
            assert_is_value(
                args, make_simple_sequence(tuple, [TypedValue(int), TypedValue(str)])
            )

        def capybara():
            f(1, "x")
            f(1)  # E: incompatible_call
            f(1, 1)  # E: incompatible_argument

    @assert_passes()
    def test_kwargs(self):
        from typing_extensions import NotRequired, Required, TypedDict, Unpack

        class TD(TypedDict):
            a: NotRequired[int]
            b: Required[str]

        def capybara(**kwargs: Unpack[TD]):
            assert_is_value(
                kwargs,
                TypedDictValue(
                    {"a": (False, TypedValue(int)), "b": (True, TypedValue(str))}
                ),
            )

        def caller():
            capybara(a=1, b="x")
            capybara(b="x")
            capybara()  # E: incompatible_call
            capybara(a="x", b="x")  # E: incompatible_argument
            capybara(a=1, b="x", c=3)  # E: incompatible_call

    @assert_passes()
    def test_invalid(self):
        from typing_extensions import Unpack

        def unpack_that_int(*args: Unpack[int]) -> None:  # E: invalid_annotation
            assert_is_value(args, AnyValue(AnySource.error))

        def bad_kwargs(**kwargs: Unpack[None]) -> None:  # E: invalid_annotation
            assert_is_value(kwargs, AnyValue(AnySource.error))


class TestTooManyPosArgs(TestNameCheckVisitorBase):
    def test_basic(self):
        self.assert_is_changed(
            """
            def f(a, b, c, d, e, f, g, h, i, j, k):
                pass

            def capybara():
                f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
            """,
            """
            def f(a, b, c, d, e, f, g, h, i, j, k):
                pass

            def capybara():
                f(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10, k=11)
            """,
        )

    def test_method(self):
        self.assert_is_changed(
            """
            class X:
                def f(self, a, b, c, d, e, f, g, h, i, j, k):
                    pass

            def capybara(x: X):
                x.f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
            """,
            """
            class X:
                def f(self, a, b, c, d, e, f, g, h, i, j, k):
                    pass

            def capybara(x: X):
                x.f(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10, k=11)
            """,
        )
