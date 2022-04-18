# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .tests import make_simple_sequence
from .value import (
    AnySource,
    AnyValue,
    assert_is_value,
    CallableValue,
    DictIncompleteValue,
    GenericValue,
    KnownValue,
    KVPair,
    MultiValuedValue,
    NO_RETURN_VALUE,
    SequenceValue,
    SubclassValue,
    TypedValue,
)


class TestSuperCall(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        # there was a bug where we would insert the 'self' argument twice for super methods
        class Cachiyacuy(object):
            def eat_food(self):
                pass

        class Acouchy(Cachiyacuy):
            def do_it(self):
                return self.eat_food()

            def eat_food(self):
                super(Acouchy, self).eat_food()

    @assert_passes()
    def test_super_no_args(self):
        class Canaanimys:
            def __init__(self, a, b):
                super().__init__()

    @assert_passes()
    def test_super_no_args_wrong_args(self):
        class Gaudeamus:
            def eat(self):
                pass

        class Canaanimys(Gaudeamus):
            def eat(self, grass=None):
                super(Canaanimys, self).eat(grass)  # E: incompatible_call

    @assert_passes()
    def test_super_no_args_wrong_args_classmethod(self):
        class Gaudeamus:
            @classmethod
            def eat(cls):
                pass

        class Canaanimys(Gaudeamus):
            @classmethod
            def eat(cls, grass):
                super().eat(grass)  # E: incompatible_call

    @assert_passes()
    def test_super_no_args_in_comprehension(self):
        class Canaanimys:
            def __init__(self, a, b):
                self.x = [super().__init__() for _ in range(1)]  # E: bad_super_call

    @assert_passes()
    def test_super_no_args_in_gen_exp(self):
        class Canaanimys:
            def __init__(self, a, b):
                self.x = (super().__init__() for _ in range(1))  # E: bad_super_call

    @assert_passes()
    def test_super_no_args_in_nested_function(self):
        class Canaanimys:
            def __init__(self, a, b):
                def nested():
                    self.x = super().__init__()  # E: bad_super_call

                nested()

    @assert_passes()
    def test_super_init_subclass(self):
        class Pithanotomys:
            def __init_subclass__(self):
                super().__init_subclass__()

    @assert_passes()
    def test_good_super_call(self):
        from pyanalyze.tests import PropertyObject, wrap

        @wrap
        class Tainotherium(PropertyObject):
            def non_async_method(self):
                super(Tainotherium.base, self).non_async_method()

    @assert_passes()
    def test_bad_super_call(self):
        from pyanalyze.tests import PropertyObject, wrap

        @wrap
        class Tainotherium2(PropertyObject):
            def non_async_method(self):
                super(Tainotherium2, self).non_async_method()  # E: bad_super_call

    @assert_passes()
    def test_first_arg_is_base(self):
        class Base1(object):
            def method(self):
                pass

        class Base2(Base1):
            def method(self):
                pass

        class Child(Base2):
            def method(self):
                super(Base2, self).method()  # E: bad_super_call

    @assert_passes()
    def test_bad_super_call_classmethod(self):
        from pyanalyze.tests import PropertyObject, wrap

        @wrap
        class Tainotherium3(PropertyObject):
            @classmethod
            def no_args_classmethod(cls):
                super(Tainotherium3, cls).no_args_classmethod()  # E: bad_super_call

    @assert_passes()
    def test_super_attribute(self):
        class MotherCapybara(object):
            def __init__(self, grass):
                pass

        class ChildCapybara(MotherCapybara):
            def __init__(self):
                super(ChildCapybara, self).__init__()  # E: incompatible_call

    @assert_passes()
    def test_undefined_super_attribute(self):
        class MotherCapybara(object):
            pass

        class ChildCapybara(MotherCapybara):
            @classmethod
            def toggle(cls):
                super(ChildCapybara, cls).toggle()  # E: undefined_attribute

    @assert_passes()
    def test_metaclass(self):
        class CapybaraType(type):
            def __init__(self, name, bases, attrs):
                super(CapybaraType, self).__init__(name, bases, attrs)

        class Capybara(metaclass=CapybaraType):
            pass

    @assert_passes()
    def test_mixin(self):
        class Base(object):
            @classmethod
            def eat(cls):
                pass

        class Mixin(object):
            @classmethod
            def eat(cls):
                super(Mixin, cls).eat()

        class Capybara(Mixin, Base):
            pass


class TestSequenceImpl(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import Sequence

        from typing_extensions import Literal

        def capybara(x, ints: Sequence[Literal[1, 2]]):
            # no arguments
            assert_is_value(set(), KnownValue(set()))
            assert_is_value(list(), KnownValue([]))

            # KnownValue
            assert_is_value(tuple([1, 2, 3]), KnownValue((1, 2, 3)))

            # Comprehensions
            one_two = MultiValuedValue([KnownValue(1), KnownValue(2)])
            assert_is_value(tuple(i for i in ints), GenericValue(tuple, [one_two]))
            assert_is_value(tuple({i: i for i in ints}), GenericValue(tuple, [one_two]))

            # SequenceValue
            assert_is_value(
                tuple([int(x)]), make_simple_sequence(tuple, [TypedValue(int)])
            )

            # fallback
            assert_is_value(
                tuple(x), GenericValue(tuple, [AnyValue(AnySource.generic_argument)])
            )

            # argument that is iterable but does not have __iter__
            assert_is_value(tuple(str(x)), GenericValue(tuple, [TypedValue(str)]))

    @assert_passes()
    def test_union(self):
        from typing import Sequence, Union

        from typing_extensions import Never

        def capybara(x: Union[Sequence[int], Sequence[str]], never: Never):
            assert_is_value(
                tuple(x),
                GenericValue(tuple, [TypedValue(int)])
                | GenericValue(tuple, [TypedValue(str)]),
            )
            assert_is_value(tuple(never), GenericValue(tuple, [NO_RETURN_VALUE]))

    @assert_passes()
    def test_not_iterable(self):
        def capybara(x):
            tuple(3)  # E: unsupported_operation
            tuple(int(x))  # E: unsupported_operation


class TestFormat(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        def capybara():
            assert_is_value("{}".format(0), TypedValue(str))
            assert_is_value("{x}".format(x=0), TypedValue(str))
            assert_is_value("{} {x.imag!r:.2d}".format(0, x=0), TypedValue(str))
            assert_is_value("{x[0]} {y[x]}".format(x=[0], y={"x": 0}), TypedValue(str))
            assert_is_value("{{X}} {}".format(0), TypedValue(str))
            assert_is_value("{0:.{1:d}e}".format(0, 1), TypedValue(str))
            assert_is_value("{:<{width}}".format("", width=1), TypedValue(str))

    @assert_passes()
    def test_errors(self):
        def out_of_range_implicit():
            "{} {}".format(0)  # E: incompatible_call

        def out_of_range_numbered():
            "{0} {1}".format(0)  # E: incompatible_call

        def out_of_range_named():
            "{x}".format(y=3)  # E: incompatible_call

        def unused_numbered():
            "{}".format(0, 1)  # E: incompatible_call

        def unused_names():
            "{x}".format(x=0, y=1)  # E: incompatible_call

    @assert_passes()
    def test_union(self):
        def capybara(cond):
            if cond:
                template = "{a} {b}"
            else:
                template = "{a} {b} {c}"
            string = template.format(a="a", b="b", c="c")
            assert_is_value(string, TypedValue(str))


class TestTypeMethods(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        class Capybara(object):
            def __init__(self, name):
                pass

            def foo(self):
                print(Capybara.__subclasses__())


class TestEncodeDecode(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def capybara(s: str, b: bytes):
            assert_is_value(s.encode("utf-8"), TypedValue(bytes))
            assert_is_value(b.decode("utf-8"), TypedValue(str))

    @assert_passes()
    def test_encode_wrong_type(self):
        def capybara():
            # TODO this should produce only one error
            "".encode(42)  # E: incompatible_call  # E: incompatible_argument

    @assert_passes()
    def test_decode_wrong_type(self):
        def capybara():
            b"".decode(42)  # E: incompatible_call  # E: incompatible_argument


class TestLen(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def capybara(x):
            assert_is_value(len("a"), KnownValue(1))
            assert_is_value(len(list(x)), TypedValue(int))

            # if we don't know the type, there should be no error
            len(x)

    @assert_passes()
    def test_narrowing(self):
        def capybara(cond):
            lst = () if cond else (1,)
            assert_is_value(lst, MultiValuedValue([KnownValue(()), KnownValue((1,))]))
            if len(lst) == 1:
                assert_is_value(lst, KnownValue((1,)))
            else:
                assert_is_value(lst, KnownValue(()))
            if len(lst) > 0:
                assert_is_value(lst, KnownValue((1,)))
            else:
                assert_is_value(lst, KnownValue(()))

    @assert_passes()
    def test_wrong_type(self):
        def capybara():
            len(3)  # E: incompatible_argument


class TestBool(TestNameCheckVisitorBase):
    @assert_passes()
    def test_return_value(self):
        def capybara(x):
            assert_is_value(bool(), KnownValue(False))
            assert_is_value(bool(x + 1), TypedValue(bool))

    @assert_passes()
    def test_constraint(self):
        from typing import Optional

        def capybara(x: Optional[int]):
            if bool(x):
                assert_is_value(x, TypedValue(int))
            else:
                assert_is_value(x, TypedValue(int) | KnownValue(None))


class TestCast(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import cast, List

        def capybara():
            assert_is_value(cast(str, 1), TypedValue(str))
            assert_is_value(cast("str", 1), TypedValue(str))
            assert_is_value(cast("List[str]", 1), GenericValue(list, [TypedValue(str)]))

    @assert_passes()
    def test_undefined_name(self):
        from typing import cast, List

        def capybara():
            cast("List[fail]", 1)  # E: undefined_name


class TestSubclasses(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        class Parent:
            pass

        class Child(Parent):
            pass

        def capybara(typ: type):
            assert_is_value(
                typ.__subclasses__(), GenericValue(list, [TypedValue(type)])
            )
            assert_is_value(Parent.__subclasses__(), KnownValue([Child]))


class TestGenericMutators(TestNameCheckVisitorBase):
    @assert_passes()
    def test_list_append(self):
        from typing import List

        def capybara(x: int):
            lst = [x]
            assert_is_value(lst, make_simple_sequence(list, [TypedValue(int)]))
            lst.append(1)
            assert_is_value(
                lst, make_simple_sequence(list, [TypedValue(int), KnownValue(1)])
            )

            lst2: List[str] = ["x"]
            assert_is_value(lst2, GenericValue(list, [TypedValue(str)]))
            lst2.append("y")
            assert_is_value(lst2, GenericValue(list, [TypedValue(str)]))

            lst3 = ["x"]
            assert_is_value(lst3, KnownValue(["x"]))
            lst3.append(3)
            assert_is_value(lst3, KnownValue(["x", 3]))

    @assert_passes()
    def test_list_append_pos_only(self):
        from typing import List

        def capybara(lst: List[int]) -> None:
            lst.append(object=42)  # E: incompatible_call

    @assert_passes()
    def test_list_append_wrong_type(self):
        from typing import List

        def capybara():
            lst: List[str] = ["x"]
            assert_is_value(lst, GenericValue(list, [TypedValue(str)]))
            lst.append(1)  # E: incompatible_argument

            double_lst: List[List[int]] = [[42]]
            assert_is_value(
                double_lst, GenericValue(list, [GenericValue(list, [TypedValue(int)])])
            )
            double_lst[0].append("x")  # E: incompatible_argument

    @assert_passes()
    def test_set_add(self):
        from typing import Set

        def capybara(x: int):
            s = {x}
            assert_is_value(s, make_simple_sequence(set, [TypedValue(int)]))
            s.add(1)
            assert_is_value(
                s, make_simple_sequence(set, [TypedValue(int), KnownValue(1)])
            )

            s2: Set[str] = {"x"}
            assert_is_value(s2, GenericValue(set, [TypedValue(str)]))
            s2.add("y")
            assert_is_value(s2, GenericValue(set, [TypedValue(str)]))

    @assert_passes()
    def test_list_add(self):
        from typing import List

        def capybara(x: int, y: str) -> None:
            assert_is_value(
                [x] + [y],
                make_simple_sequence(list, [TypedValue(int), TypedValue(str)]),
            )
            assert_is_value(
                [x] + [1], make_simple_sequence(list, [TypedValue(int), KnownValue(1)])
            )
            left: List[int] = []
            right: List[str] = []
            assert_is_value(
                left + right,
                GenericValue(
                    list, [MultiValuedValue([TypedValue(int), TypedValue(str)])]
                ),
            )
            assert_is_value(left + left, GenericValue(list, [TypedValue(int)]))

            union_list1 = left if x else right
            union_list2 = left if y else right
            assert_is_value(
                # need to call list.__add__ directly because we just give up on unions
                # in the binop implementation
                list.__add__(union_list1, union_list2),
                MultiValuedValue(
                    [
                        GenericValue(list, [TypedValue(int)]),
                        GenericValue(
                            list, [MultiValuedValue([TypedValue(int), TypedValue(str)])]
                        ),
                        GenericValue(
                            list, [MultiValuedValue([TypedValue(str), TypedValue(int)])]
                        ),
                        GenericValue(list, [TypedValue(str)]),
                    ]
                ),
            )

    @assert_passes()
    def test_list_extend(self):
        from typing import List

        def capybara(x: int, y: str) -> None:
            lst = [x]
            assert_is_value(lst, make_simple_sequence(list, [TypedValue(int)]))
            lst.extend([y])
            assert_is_value(
                lst, make_simple_sequence(list, [TypedValue(int), TypedValue(str)])
            )
            lst.extend({float(1.0)})
            assert_is_value(
                lst,
                make_simple_sequence(
                    list, [TypedValue(int), TypedValue(str), TypedValue(float)]
                ),
            )

            lst: List[int] = [3]
            assert_is_value(lst, GenericValue(list, [TypedValue(int)]))
            lst.extend([x])
            assert_is_value(lst, GenericValue(list, [TypedValue(int)]))

    @assert_passes()
    def test_list_iadd(self):
        from typing import List

        def capybara(x: int, y: str) -> None:
            lst = [x]
            assert_is_value(lst, make_simple_sequence(list, [TypedValue(int)]))
            lst += [y]
            assert_is_value(
                lst, make_simple_sequence(list, [TypedValue(int), TypedValue(str)])
            )
            lst += {float(1.0)}
            assert_is_value(
                lst,
                make_simple_sequence(
                    list, [TypedValue(int), TypedValue(str), TypedValue(float)]
                ),
            )

            lst2: List[int] = [3]
            assert_is_value(lst2, GenericValue(list, [TypedValue(int)]))
            lst2 += [x]
            assert_is_value(lst2, GenericValue(list, [TypedValue(int)]))

    @assert_passes()
    def test_list_iadd_never(self):
        def render_feedback_text():
            z = []

            detail_text = None
            if detail_text:
                assert_is_value(detail_text, NO_RETURN_VALUE)
                z += detail_text

            return z

    @assert_passes()
    def test_weak_value(self):
        from typing import List

        from typing_extensions import Literal

        def func() -> List[Literal["c", "d"]]:
            return ["d", "c"]

        def capybara() -> None:
            lst = ["a", "b"]
            assert_is_value(lst, KnownValue(["a", "b"]))
            lst.extend(func())
            assert_is_value(
                lst,
                SequenceValue(
                    list,
                    [
                        (False, KnownValue("a")),
                        (False, KnownValue("b")),
                        (True, KnownValue("c") | KnownValue("d")),
                    ],
                ),
            )
            lst.extend(["e"])
            assert_is_value(
                lst,
                SequenceValue(
                    list,
                    [
                        (False, KnownValue("a")),
                        (False, KnownValue("b")),
                        (True, KnownValue("c") | KnownValue("d")),
                        (False, KnownValue("e")),
                    ],
                ),
            )
            lst.append("f")
            assert_is_value(
                lst,
                SequenceValue(
                    list,
                    [
                        (False, KnownValue("a")),
                        (False, KnownValue("b")),
                        (True, KnownValue("c") | KnownValue("d")),
                        (False, KnownValue("e")),
                        (False, KnownValue("f")),
                    ],
                ),
            )

    @assert_passes()
    def test_starred_weak(self):
        from typing import List

        from typing_extensions import Literal

        def capybara(arg) -> None:
            lst1: List[Literal["a"]] = ["a" for _ in arg]
            lst2 = [*lst1, "b"]
            assert_is_value(
                lst2,
                SequenceValue(
                    list, [(True, KnownValue("a")), (False, KnownValue("b"))]
                ),
            )
            lst2.append("c")
            assert_is_value(
                lst2,
                SequenceValue(
                    list,
                    [
                        (True, KnownValue("a")),
                        (False, KnownValue("b")),
                        (False, KnownValue("c")),
                    ],
                ),
            )

    @assert_passes()
    def test_list_extend_wrong_type(self):
        from typing import List

        def capybara():
            lst: List[int] = [3]
            lst.extend([str(3)])  # E: incompatible_argument

    @assert_passes()
    def test_list_extend_union(self):
        def capybara(cond):
            if cond:
                lst = [1 for _ in cond]
            else:
                lst = [2 for _ in cond]
            assert_is_value(
                lst,
                SequenceValue(list, [(True, KnownValue(1))])
                | SequenceValue(list, [(True, KnownValue(2))]),
            )
            lst.extend([3, 4])

            assert_is_value(
                lst,
                SequenceValue(
                    list,
                    [
                        (True, KnownValue(1)),
                        (False, KnownValue(3)),
                        (False, KnownValue(4)),
                    ],
                )
                | SequenceValue(
                    list,
                    [
                        (True, KnownValue(2)),
                        (False, KnownValue(3)),
                        (False, KnownValue(4)),
                    ],
                ),
            )

    @assert_passes()
    def test_dict_get(self):
        from typing import Dict

        from typing_extensions import NotRequired, TypedDict

        class TD(TypedDict):
            a: int
            b: str
            c: NotRequired[str]

        def capybara(td: TD, s: str, d: Dict[str, int], untyped: dict):
            assert_is_value(td.get("a"), TypedValue(int))
            assert_is_value(td.get("c"), TypedValue(str) | KnownValue(None))
            assert_is_value(td.get("c", 1), TypedValue(str) | KnownValue(1))
            td.get(1)  # E: invalid_typeddict_key

            known = {"a": "b"}
            assert_is_value(known.get("a"), KnownValue("b") | KnownValue(None))
            assert_is_value(known.get("b", 1), KnownValue(1))
            assert_is_value(known.get(s), KnownValue("b") | KnownValue(None))

            incomplete = {**td, "b": 1, "d": s}
            assert_is_value(incomplete.get("a"), TypedValue(int) | KnownValue(None))
            assert_is_value(incomplete.get("b"), KnownValue(1) | KnownValue(None))
            assert_is_value(incomplete.get("d"), TypedValue(str) | KnownValue(None))
            assert_is_value(incomplete.get("e"), KnownValue(None))

            assert_is_value(d.get("x"), TypedValue(int) | KnownValue(None))
            assert_is_value(d.get(s), TypedValue(int) | KnownValue(None))
            d.get(1)  # E: incompatible_argument

            untyped.get([])  # E: unhashable_key

    @assert_passes()
    def test_setdefault(self):
        from typing import Dict, Sequence

        from typing_extensions import TypedDict

        class TD(TypedDict):
            a: int
            b: str

        def typeddict(td: TD):
            td.setdefault({})  # E: unhashable_key
            td.setdefault(0)  # E: invalid_typeddict_key
            td.setdefault("c")  # E: invalid_typeddict_key
            td.setdefault("a", "s")  # E: incompatible_argument
            assert_is_value(td.setdefault("b", "x"), TypedValue(str))

        def dict_incomplete_value():
            incomplete_value = {"a": str(TD)}
            assert_is_value(
                incomplete_value,
                DictIncompleteValue(dict, [KVPair(KnownValue("a"), TypedValue(str))]),
            )
            assert_is_value(incomplete_value.setdefault("b"), KnownValue(None))
            assert_is_value(
                incomplete_value,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue("a"), TypedValue(str)),
                        KVPair(KnownValue("b"), KnownValue(None)),
                    ],
                ),
            )
            assert_is_value(
                incomplete_value.setdefault("a"),
                MultiValuedValue([KnownValue(None), TypedValue(str)]),
            )
            assert_is_value(
                incomplete_value,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue("a"), TypedValue(str)),
                        KVPair(KnownValue("b"), KnownValue(None)),
                        KVPair(KnownValue("a"), KnownValue(None), is_required=False),
                    ],
                ),
            )

        def weak_typed(ints: Sequence[int]):
            weak_dict = {i: str(i) for i in ints}
            assert_is_value(
                weak_dict,
                DictIncompleteValue(
                    dict, [KVPair(TypedValue(int), TypedValue(str), is_many=True)]
                ),
            )
            assert_is_value(weak_dict.setdefault(3, str(TD)), TypedValue(str))

            assert_is_value(
                weak_dict,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(TypedValue(int), TypedValue(str), is_many=True),
                        KVPair(KnownValue(3), TypedValue(str), is_required=False),
                    ],
                ),
            )
            assert_is_value(weak_dict.setdefault(3), TypedValue(str) | KnownValue(None))
            assert_is_value(
                weak_dict,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(TypedValue(int), TypedValue(str), is_many=True),
                        KVPair(KnownValue(3), TypedValue(str), is_required=False),
                        KVPair(KnownValue(3), KnownValue(None), is_required=False),
                    ],
                ),
            )

        def strong_typed(strong_dict: Dict[int, str]):
            expected = GenericValue(dict, [TypedValue(int), TypedValue(str)])
            assert_is_value(strong_dict, expected)
            assert_is_value(strong_dict.setdefault(3, str(TD)), TypedValue(str))
            assert_is_value(strong_dict, expected)
            assert_is_value(
                strong_dict.setdefault(3),
                MultiValuedValue([TypedValue(str), KnownValue(None)]),
            )
            assert_is_value(strong_dict, expected)

    @assert_passes()
    def test_dict_pop(self):
        from typing import Dict

        from typing_extensions import NotRequired, TypedDict

        class TD(TypedDict):
            a: int
            b: NotRequired[str]

        def typeddict(td: TD):
            td.pop({})  # E: unhashable_key
            td.pop(0)  # E: invalid_typeddict_key
            td.pop("c")  # E: invalid_typeddict_key
            val = td.pop("a", "s")  # E: incompatible_argument
            assert_is_value(val, TypedValue(int) | KnownValue("s"))
            assert_is_value(td.pop("b", "x"), TypedValue(str) | KnownValue("x"))

        def dict_incomplete_value():
            incomplete_value = {"a": str(TD), "c": 42}
            assert_is_value(
                incomplete_value,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue("a"), TypedValue(str)),
                        KVPair(KnownValue("c"), KnownValue(42)),
                    ],
                ),
            )
            incomplete_value.pop("b")  # E: incompatible_argument
            assert_is_value(incomplete_value.pop("a"), TypedValue(str))
            assert_is_value(
                incomplete_value,
                DictIncompleteValue(dict, [KVPair(KnownValue("c"), KnownValue(42))]),
            )
            assert_is_value(
                incomplete_value.pop("c", 1), KnownValue(42) | KnownValue(1)
            )
            assert_is_value(incomplete_value, DictIncompleteValue(dict, []))

        def strong_typed(strong_dict: Dict[int, str]):
            expected = GenericValue(dict, [TypedValue(int), TypedValue(str)])
            assert_is_value(strong_dict, expected)
            assert_is_value(strong_dict.pop(3, 42), TypedValue(str) | KnownValue(42))
            assert_is_value(strong_dict, expected)
            assert_is_value(strong_dict.pop(3), TypedValue(str))
            assert_is_value(strong_dict, expected)

    @assert_passes()
    def test_dict_pop_union(self):
        def capybara(cond):
            d = {"a": 1, "b": 2}
            if cond:
                d["c"] = 3
            else:
                d["d"] = 4

            assert_is_value(d.pop("a"), KnownValue(1))

    @assert_passes()
    def test_dict_update(self):
        def capybara():
            d1 = {}
            d1.update({})
            d2 = {}
            d2.update(a=3, b=4)
            assert_is_value(
                d2,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue("a"), KnownValue(3)),
                        KVPair(KnownValue("b"), KnownValue(4)),
                    ],
                ),
            )
            d2.update([("a", 4), ("b", 5)])
            assert_is_value(
                d2,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue("a"), KnownValue(3)),
                        KVPair(KnownValue("b"), KnownValue(4)),
                        KVPair(KnownValue("a"), KnownValue(4)),
                        KVPair(KnownValue("b"), KnownValue(5)),
                    ],
                ),
            )

    @assert_passes()
    def test_copy_and_update(self):
        from typing import Dict

        def capybara():
            d1: Dict[str, int] = {"x": 1}
            d1_val = GenericValue(dict, [TypedValue(str), TypedValue(int)])
            assert_is_value(d1, d1_val)
            d1[1] = 3  # E: incompatible_argument
            d2 = d1.copy()
            assert_is_value(
                d2,
                DictIncompleteValue(
                    dict, [KVPair(TypedValue(str), TypedValue(int), is_many=True)]
                ),
            )
            d2[1] = 3
            assert_is_value(
                d2,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(TypedValue(str), TypedValue(int), is_many=True),
                        KVPair(KnownValue(1), KnownValue(3)),
                    ],
                ),
            )


class TestSequenceGetItem(TestNameCheckVisitorBase):
    @assert_passes()
    def test_list(self):
        from typing import List

        def capybara(lst: List[int], i: int, s: slice, unannotated) -> None:
            assert_is_value(lst[0], TypedValue(int))
            assert_is_value(lst[-1], TypedValue(int))
            assert_is_value(lst[:1], GenericValue(list, [TypedValue(int)]))
            assert_is_value(lst[i], TypedValue(int))
            assert_is_value(lst[s], GenericValue(list, [TypedValue(int)]))
            assert_is_value(lst[unannotated], AnyValue(AnySource.from_another))

            empty = []
            assert_is_value(empty[0], AnyValue(AnySource.unreachable))
            assert_is_value(empty[1:], KnownValue([]))
            assert_is_value(empty[i], AnyValue(AnySource.unreachable))
            assert_is_value(empty[s], SequenceValue(list, []))
            assert_is_value(empty[unannotated], AnyValue(AnySource.from_another))

            known = [1, 2]
            assert_is_value(known[0], KnownValue(1))
            assert_is_value(known[-1], KnownValue(2))
            assert_is_value(known[-5], KnownValue(1) | KnownValue(2))
            assert_is_value(known[1:], KnownValue([2]))
            assert_is_value(known[::-1], KnownValue([2, 1]))
            assert_is_value(known[i], KnownValue(1) | KnownValue(2))
            assert_is_value(
                known[s], make_simple_sequence(list, [KnownValue(1), KnownValue(2)])
            )
            assert_is_value(known[unannotated], AnyValue(AnySource.from_another))

    @assert_passes()
    def test_tuple(self):
        from typing import Tuple

        def capybara(tpl: Tuple[int, ...], i: int, s: slice, unannotated) -> None:
            assert_is_value(tpl[0], TypedValue(int))
            assert_is_value(tpl[-1], TypedValue(int))
            assert_is_value(tpl[:1], GenericValue(tuple, [TypedValue(int)]))
            assert_is_value(tpl[:], GenericValue(tuple, [TypedValue(int)]))
            assert_is_value(tpl[i], TypedValue(int))
            assert_is_value(tpl[s], GenericValue(tuple, [TypedValue(int)]))
            assert_is_value(tpl[unannotated], AnyValue(AnySource.from_another))

            empty = ()
            assert_is_value(empty[0], AnyValue(AnySource.error))  # E: incompatible_call
            assert_is_value(empty[1:], KnownValue(()))
            assert_is_value(empty[i], AnyValue(AnySource.unreachable))
            assert_is_value(empty[s], SequenceValue(tuple, []))
            assert_is_value(empty[unannotated], AnyValue(AnySource.from_another))

            known = (1, 2)
            assert_is_value(known[0], KnownValue(1))
            assert_is_value(known[-1], KnownValue(2))
            assert_is_value(
                known[-5], AnyValue(AnySource.error)  # E: incompatible_call
            )
            assert_is_value(known[1:], KnownValue((2,)))
            assert_is_value(known[::-1], KnownValue((2, 1)))
            assert_is_value(known[i], KnownValue(1) | KnownValue(2))
            assert_is_value(
                known[s], make_simple_sequence(tuple, [KnownValue(1), KnownValue(2)])
            )
            assert_is_value(known[unannotated], AnyValue(AnySource.from_another))

    @assert_passes()
    def test_list_index(self):
        def capybara(x):
            lst = ["a", "b", int(x)]
            assert_is_value(lst[0], KnownValue("a"))
            assert_is_value(lst[2], TypedValue(int))
            assert_is_value(lst[-2], KnownValue("b"))
            assert_is_value(lst[5], KnownValue("a") | KnownValue("b") | TypedValue(int))

    @assert_passes()
    def test_tuple_index(self):
        def capybara(x):
            tpl = ("a", "b", int(x))
            assert_is_value(tpl[0], KnownValue("a"))
            assert_is_value(tpl[2], TypedValue(int))
            assert_is_value(tpl[-2], KnownValue("b"))
            assert_is_value(tpl[5], AnyValue(AnySource.error))  # E: incompatible_call

    @assert_passes()
    def test_tuple_annotation(self):
        from typing import Tuple

        def capybara(tpl: Tuple[int, str, float]) -> None:
            assert_is_value(tpl[0], TypedValue(int))
            assert_is_value(tpl[-2], TypedValue(str))
            assert_is_value(tpl[2], TypedValue(float))

    @assert_passes()
    def test_list_in_lambda(self):
        from typing import List

        def capybara(words: List[str]):
            sorted_indexes = sorted(range(len(words)), key=lambda i: words[i])
            return sorted_indexes

    @assert_passes()
    def test_subclasses(self):
        import time

        class MyList(list):
            pass

        class MyTuple(tuple):
            pass

        def capybara(t: time.struct_time, ml: MyList, mt: MyTuple):
            assert_is_value(t[0], TypedValue(int))
            assert_is_value(t[:], TypedValue(tuple))
            assert_is_value(t[:6], TypedValue(tuple))

            assert_is_value(ml[0], AnyValue(AnySource.generic_argument))
            assert_is_value(ml[:], TypedValue(list))
            assert_is_value(mt[0], AnyValue(AnySource.generic_argument))
            assert_is_value(mt[:], TypedValue(tuple))


class TestDictGetItem(TestNameCheckVisitorBase):
    @assert_passes()
    def test_unhashable(self):
        def capybara():
            d = {}
            d[{}]  # E: unhashable_key

    @assert_passes()
    def test_invalid_typeddict_key(self):
        from typing_extensions import TypedDict

        class TD(TypedDict):
            a: int

        def capybara(td: TD):
            td[1]  # E: invalid_typeddict_key

    @assert_passes()
    def test_incomplete_value(self):
        def capybara(a: int, unresolved):
            incomplete_value = {a: 1, "b": 2, "c": "s"}
            assert_is_value(
                incomplete_value,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(TypedValue(int), KnownValue(1)),
                        KVPair(KnownValue("b"), KnownValue(2)),
                        KVPair(KnownValue("c"), KnownValue("s")),
                    ],
                ),
            )

            assert_is_value(incomplete_value["b"], KnownValue(2))
            assert_is_value(incomplete_value[1], KnownValue(1))
            assert_is_value(
                incomplete_value[unresolved],
                MultiValuedValue([KnownValue(1), KnownValue(2), KnownValue("s")]),
            )
            # unknown key
            assert_is_value(incomplete_value["other string"], AnyValue(AnySource.error))

            # MultiValuedValue
            key = "b" if unresolved else "c"
            assert_is_value(
                incomplete_value[key],
                MultiValuedValue([KnownValue(2), KnownValue("s")]),
            )

    @assert_passes()
    def test_complex_incomplete(self):
        from typing import Sequence

        from typing_extensions import NotRequired, TypedDict

        class TD(TypedDict):
            a: float
            b: NotRequired[bool]

        def capybara(i: int, seq: Sequence[int], td: TD, s: str):
            d1 = {"a": i, "b": i + 1}
            d2 = {i: 1 for i in seq}
            d3 = {"a": 1, **d1, "b": 2, **d2}
            assert_is_value(
                d3,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue("a"), KnownValue(1)),
                        KVPair(KnownValue("a"), TypedValue(int)),
                        KVPair(KnownValue("b"), TypedValue(int)),
                        KVPair(KnownValue("b"), KnownValue(2)),
                        KVPair(TypedValue(int), KnownValue(1), is_many=True),
                    ],
                ),
            )
            assert_is_value(d3[1], KnownValue(1))
            assert_is_value(d3["a"], TypedValue(int))
            assert_is_value(d3["b"], KnownValue(2))
            assert_is_value(d3[s], TypedValue(int) | KnownValue(2))

            d4 = {**d3, **td}
            assert_is_value(d4[1], KnownValue(1))
            assert_is_value(d4["a"], TypedValue(float))
            assert_is_value(d4["b"], KnownValue(2) | TypedValue(bool))
            assert_is_value(d4[s], TypedValue(float) | KnownValue(2) | TypedValue(bool))

    @assert_passes()
    def test(self):
        from typing import Dict, Generic, TypeVar

        from typing_extensions import TypedDict

        K = TypeVar("K")
        V = TypeVar("V")

        class ReversedDict(Generic[V, K], Dict[K, V]):
            pass

        class NormalDict(Generic[K, V], Dict[K, V]):
            pass

        class TD(TypedDict):
            a: int

        def capybara(
            td: TD,
            dct: Dict[str, int],
            rev: ReversedDict[str, int],
            nd: NormalDict[int, str],
            untyped: dict,
        ):
            d = {1: 2}
            assert_is_value(d[1], KnownValue(2))
            assert_is_value(td["a"], TypedValue(int))

            assert_is_value(dct["key"], TypedValue(int))
            assert_is_value(nd[1], TypedValue(str))
            assert_is_value(rev[1], TypedValue(str))

            untyped[[]]  # E: unhashable_key
            dct[1]  # E: incompatible_argument

    @assert_passes()
    def test_type_as_key(self):
        from typing import Type

        def capybara(d: dict, t: Type[int]):
            d[int]
            d[t]


class TestDictSetItem(TestNameCheckVisitorBase):
    @assert_passes()
    def test_typeddict_setitem_valid(self):
        from typing_extensions import TypedDict

        class TD(TypedDict):
            x: int

        def capybara(td: TD) -> None:
            td["x"] = 42

    @assert_passes()
    def test_typeddict_non_literal_key(self):
        from typing_extensions import TypedDict

        class TD(TypedDict):
            x: int

        def capybara(td: TD) -> None:
            td[41] = 42  # E: invalid_typeddict_key

    @assert_passes()
    def test_typeddict_unrecognized_key(self):
        from typing_extensions import TypedDict

        class TD(TypedDict):
            x: int

        def capybara(td: TD) -> None:
            td["y"] = 42  # E: invalid_typeddict_key

    @assert_passes()
    def test_typeddict_bad_value(self):
        from typing_extensions import TypedDict

        class TD(TypedDict):
            x: int

        def capybara(td: TD) -> None:
            td["x"] = "y"  # E: incompatible_argument

    @assert_passes()
    def test_incomplete_value(self):
        def capybara(x: int, y: str) -> None:
            dct = {}
            assert_is_value(dct, KnownValue({}))
            dct["x"] = x
            assert_is_value(
                dct,
                DictIncompleteValue(dict, [KVPair(KnownValue("x"), TypedValue(int))]),
            )
            dct[y] = "x"
            assert_is_value(
                dct,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue("x"), TypedValue(int)),
                        KVPair(TypedValue(str), KnownValue("x")),
                    ],
                ),
            )

    @assert_passes()
    def test_bad_key_type(self):
        from typing import Dict

        def capybara(untyped: dict) -> None:
            dct: Dict[str, int] = {}
            dct[1] = 1  # E: incompatible_argument

            untyped[[]] = 1  # E: unhashable_key

    @assert_passes()
    def test_bad_value_type(self):
        from typing import Dict

        def capybara() -> None:
            dct: Dict[str, int] = {}
            dct["1"] = "1"  # E: incompatible_argument

    @assert_passes()
    def test_weak(self):
        def capybara(arg):
            dct = {int(k): 1 for k in arg}
            assert_is_value(
                dct,
                DictIncompleteValue(
                    dict, [KVPair(TypedValue(int), KnownValue(1), is_many=True)]
                ),
            )
            dct["x"] = "y"
            assert_is_value(
                dct,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(TypedValue(int), KnownValue(1), is_many=True),
                        KVPair(KnownValue("x"), KnownValue("y")),
                    ],
                ),
            )


class TestIssubclass(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        def capybara(x: type, y):
            assert_is_value(x, TypedValue(type))
            if issubclass(x, str):
                assert_is_value(x, SubclassValue(TypedValue(str)))
            if issubclass(y, (int, str)):
                assert_is_value(
                    y,
                    MultiValuedValue(
                        [SubclassValue(TypedValue(int)), SubclassValue(TypedValue(str))]
                    ),
                )

    @assert_passes()
    def test_negative_narrowing(self) -> None:
        from typing import Type, Union

        def capybara(x: Union[Type[str], Type[int]]) -> None:
            assert_is_value(
                x, SubclassValue(TypedValue(str)) | SubclassValue(TypedValue(int))
            )
            if issubclass(x, str):
                assert_is_value(x, SubclassValue(TypedValue(str)))
            else:
                assert_is_value(x, SubclassValue(TypedValue(int)))


class TestCallableGuards(TestNameCheckVisitorBase):
    @assert_passes()
    def test_callable(self):
        from pyanalyze.signature import ANY_SIGNATURE

        def capybara(o: object) -> None:
            assert_is_value(o, TypedValue(object))
            if callable(o):
                assert_is_value(o, CallableValue(ANY_SIGNATURE))

    @assert_passes()
    def test_isfunction(self):
        import inspect
        from types import FunctionType

        def capybara(o: object) -> None:
            assert_is_value(o, TypedValue(object))
            if inspect.isfunction(o):
                assert_is_value(o, TypedValue(FunctionType))
