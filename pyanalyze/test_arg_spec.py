# static analysis: ignore
from asynq import asynq
from qcore.asserts import assert_eq
from qcore.testing import Anything
import collections.abc
from collections.abc import (
    MutableSequence,
    Sequence,
    Collection,
    Reversible,
    Set,
)
import contextlib
import time
from typing import Generic, TypeVar

from .test_config import TestConfig
from .test_name_check_visitor import (
    TestNameCheckVisitorBase,
    ConfiguredNameCheckVisitor,
)
from .test_node_visitor import assert_fails, assert_passes, skip_before
from .arg_spec import (
    ArgSpecCache,
    BoundMethodArgSpecWrapper,
    ExtendedArgSpec,
    Parameter,
    TypeshedFinder,
    is_dot_asynq_function,
)
from .error_code import ErrorCode
from .tests import l0cached_async_fn
from .value import KnownValue, TypedValue, GenericValue, TypeVarValue, UNRESOLVED_VALUE

T = TypeVar("T")


class ClassWithCall(object):
    def __init__(self, name):
        pass

    def __call__(self, arg):
        pass

    @classmethod
    def normal_classmethod(cls):
        pass

    @staticmethod
    def normal_staticmethod(arg):
        pass

    @asynq()
    def async_method(self, x):
        pass

    @asynq()
    @staticmethod
    def async_staticmethod(y):
        pass

    @asynq()
    @classmethod
    def async_classmethod(cls, z):
        pass

    @asynq(pure=True)
    @classmethod
    def pure_async_classmethod(cls, ac):
        pass

    @classmethod
    @asynq()
    def classmethod_before_async(cls, ac):
        pass


def function(capybara, hutia=3, *tucotucos, **proechimys):
    pass


@asynq()
def async_function(x, y):
    pass


def test_get_argspec():
    visitor = ConfiguredNameCheckVisitor(__file__, u"", {}, fail_after_first=False)
    config = visitor.config
    cwc_typed = TypedValue(ClassWithCall)
    cwc_self = Parameter("self", typ=cwc_typed)

    # test everything twice because calling qcore.get_original_fn has side effects
    for _ in range(2):

        # there's special logic for this in _get_argspec_from_value; TODO move that into
        # ExtendedArgSpec
        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec(
                    arguments=[cwc_self, Parameter("arg")], name="ClassWithCall"
                ),
                cwc_typed,
            ),
            visitor._get_argspec_from_value(cwc_typed, None),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("cls")]), KnownValue(ClassWithCall)
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.normal_classmethod),
        )
        assert_eq(
            ExtendedArgSpec([Parameter("arg")]),
            ArgSpecCache(config).get_argspec(ClassWithCall.normal_staticmethod),
        )

        assert_eq(
            ExtendedArgSpec(
                [Parameter("capybara"), Parameter("hutia", default_value=3)],
                starargs="tucotucos",
                kwargs="proechimys",
            ),
            ArgSpecCache(config).get_argspec(function),
        )

        assert_eq(
            ExtendedArgSpec([Parameter("x"), Parameter("y")]),
            ArgSpecCache(config).get_argspec(async_function),
        )

        assert_eq(
            ExtendedArgSpec([Parameter("x"), Parameter("y")]),
            ArgSpecCache(config).get_argspec(async_function.asynq),
        )

        instance = ClassWithCall(1)

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("self"), Parameter("x")]),
                KnownValue(instance),
            ),
            ArgSpecCache(config).get_argspec(instance.async_method),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("self"), Parameter("x")]),
                KnownValue(instance),
            ),
            ArgSpecCache(config).get_argspec(instance.async_method.asynq),
        )

        assert_eq(
            ExtendedArgSpec([Parameter("y")]),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_staticmethod),
        )

        assert_eq(
            ExtendedArgSpec([Parameter("y")]),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_staticmethod.asynq),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("cls"), Parameter("z")]),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_classmethod),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("cls"), Parameter("z")]),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.async_classmethod.asynq),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("cls"), Parameter("ac")]),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.pure_async_classmethod),
        )

        assert_eq(
            BoundMethodArgSpecWrapper(
                ExtendedArgSpec([Parameter("cls"), Parameter("ac")]),
                KnownValue(ClassWithCall),
            ),
            ArgSpecCache(config).get_argspec(ClassWithCall.classmethod_before_async),
        )


def test_is_dot_asynq_function():
    assert not is_dot_asynq_function(async_function)
    assert is_dot_asynq_function(async_function.asynq)
    assert not is_dot_asynq_function(l0cached_async_fn)
    assert is_dot_asynq_function(l0cached_async_fn.asynq)
    assert not is_dot_asynq_function(l0cached_async_fn.dirty)


class TestProperty(TestNameCheckVisitorBase):
    @assert_passes()
    def test_property(self):
        from pyanalyze.tests import PropertyObject

        def capybara(uid):
            assert_is_value(PropertyObject(uid).string_property, TypedValue(str))


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

    @assert_fails(ErrorCode.incompatible_call)
    def test_super_no_args_wrong_args(self):
        class Gaudeamus:
            def eat(self):
                pass

        class Canaanimys(Gaudeamus):
            def eat(self, grass):
                super(Canaanimys, self).eat(grass)

    @assert_fails(ErrorCode.incompatible_call)
    def test_super_no_args_wrong_args_classmethod(self):
        class Gaudeamus:
            @classmethod
            def eat(cls):
                pass

        class Canaanimys(Gaudeamus):
            @classmethod
            def eat(cls, grass):
                super().eat(grass)

    @assert_fails(ErrorCode.bad_super_call)
    def test_super_no_args_in_comprehension(self):
        class Canaanimys:
            def __init__(self, a, b):
                self.x = [super().__init__() for _ in range(1)]

    @assert_fails(ErrorCode.bad_super_call)
    def test_super_no_args_in_gen_exp(self):
        class Canaanimys:
            def __init__(self, a, b):
                self.x = (super().__init__() for _ in range(1))

    @assert_fails(ErrorCode.bad_super_call)
    def test_super_no_args_in_nested_function(self):
        class Canaanimys:
            def __init__(self, a, b):
                def nested():
                    self.x = super().__init__()

                nested()

    @assert_passes()
    def test_super_init_subclass(self):
        class Pithanotomys:
            def __init_subclass__(self):
                super().__init_subclass__()

    @assert_passes()
    def test_good_super_call(self):
        from pyanalyze.tests import wrap, PropertyObject

        @wrap
        class Tainotherium(PropertyObject):
            def non_async_method(self):
                super(Tainotherium.base, self).non_async_method()

    @assert_fails(ErrorCode.bad_super_call)
    def test_bad_super_call(self):
        from pyanalyze.tests import wrap, PropertyObject

        @wrap
        class Tainotherium2(PropertyObject):
            def non_async_method(self):
                super(Tainotherium2, self).non_async_method()

    @assert_fails(ErrorCode.bad_super_call)
    def test_first_arg_is_base(self):
        class Base1(object):
            def method(self):
                pass

        class Base2(Base1):
            def method(self):
                pass

        class Child(Base2):
            def method(self):
                super(Base2, self).method()

    @assert_fails(ErrorCode.bad_super_call)
    def test_bad_super_call_classmethod(self):
        from pyanalyze.tests import wrap, PropertyObject

        @wrap
        class Tainotherium3(PropertyObject):
            @classmethod
            def no_args_classmethod(cls):
                super(Tainotherium3, cls).no_args_classmethod()

    @assert_fails(ErrorCode.incompatible_call)
    def test_super_attribute(self):
        class MotherCapybara(object):
            def __init__(self, grass):
                pass

        class ChildCapybara(MotherCapybara):
            def __init__(self):
                super(ChildCapybara, self).__init__()

    @assert_fails(ErrorCode.undefined_attribute)
    def test_undefined_super_attribute(self):
        class MotherCapybara(object):
            pass

        class ChildCapybara(MotherCapybara):
            @classmethod
            def toggle(cls):
                super(ChildCapybara, cls).toggle()

    @assert_passes()
    def test_metaclass(self):
        import six

        class CapybaraType(type):
            def __init__(self, name, bases, attrs):
                super(CapybaraType, self).__init__(name, bases, attrs)

        class Capybara(six.with_metaclass(CapybaraType)):
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

    @assert_passes()
    def test_multi_valued(self):
        Capybara = 42

        class Capybara(object):
            pass

        C = Capybara

        def fn():
            assert_is_value(Capybara, MultiValuedValue([KnownValue(42), KnownValue(C)]))


class TestSequenceImpl(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def capybara(x):
            # no arguments
            assert_is_value(set(), KnownValue(set()))
            assert_is_value(list(), KnownValue([]))

            # KnownValue
            assert_is_value(tuple([1, 2, 3]), KnownValue((1, 2, 3)))

            # Comprehensions
            one_two = MultiValuedValue([KnownValue(1), KnownValue(2)])
            assert_is_value(tuple(i for i in (1, 2)), GenericValue(tuple, [one_two]))
            assert_is_value(
                tuple({i: i for i in (1, 2)}), GenericValue(tuple, [one_two])
            )

            # SequenceIncompleteValue
            assert_is_value(
                tuple([int(x)]), SequenceIncompleteValue(tuple, [TypedValue(int)])
            )

            # fallback
            assert_is_value(tuple(x), TypedValue(tuple))

            # argument that is iterable but does not have __iter__
            assert_is_value(tuple(str(x)), TypedValue(tuple))

    @assert_fails(ErrorCode.unsupported_operation)
    def test_tuple_known_int(self):
        def capybara(x):
            tuple(3)

    @assert_fails(ErrorCode.unsupported_operation)
    def test_tuple_typed_int(self):
        def capybara(x):
            tuple(int(x))


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

    @assert_fails(ErrorCode.incompatible_call)
    def test_out_of_range_implicit(self):
        def capybara():
            "{} {}".format(0)

    @assert_fails(ErrorCode.incompatible_call)
    def test_out_of_range_numbered(self):
        def capybara():
            "{0} {1}".format(0)

    @assert_fails(ErrorCode.incompatible_call)
    def test_out_of_range_named(self):
        def capybara():
            "{x}".format(y=3)

    @assert_fails(ErrorCode.incompatible_call)
    def test_unused_numbered(self):
        def capybara():
            "{}".format(0, 1)

    @assert_fails(ErrorCode.incompatible_call)
    def test_unused_named(self):
        def capybara():
            "{x}".format(x=0, y=1)


class TestShadowing(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def shadow_them(locals, __import__, *list, **dict):
            return (
                [int for int in list] * locals + __import__ + [v for v in dict.values()]
            )

        shadow_them(5, [1, 2], 3)


class TestTypeMethods(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        class Capybara(object):
            def __init__(self, name):
                pass

            def foo(self):
                print(Capybara.__subclasses__())


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
            assert_is_value(obj(x), UNRESOLVED_VALUE)

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
        def capybara(x):
            l = hasattr(x, "foo")
            assert_is_value(l, TypedValue(bool))

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

        uid = 1
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
    def test_kwonly_args_subclass(self):
        from pyanalyze.tests import KeywordOnlyArguments

        class Capybara(KeywordOnlyArguments):
            def __init__(self):
                pass

        def run():
            Capybara(1)

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


class TestEncodeDecode(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        import six

        def capybara():
            assert_is_value(u"".encode("utf-8"), TypedValue(bytes))
            assert_is_value(b"".decode("utf-8"), TypedValue(six.text_type))

    @assert_fails(ErrorCode.incompatible_argument)
    def test_encode_wrong_type(self):
        def capybara():
            u"".encode(42)

    @assert_fails(ErrorCode.incompatible_argument)
    def test_decode_wrong_type(self):
        def capybara():
            b"".decode(42)


class TestLen(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def capybara(x):
            assert_is_value(len("a"), TypedValue(int))
            assert_is_value(len(list(x)), TypedValue(int))

            # if we don't know the type, there should be no error
            len(x)

    @assert_fails(ErrorCode.incompatible_argument)
    def test_wrong_type(self):
        def capybara():
            len(3)


class TestCoroutines(TestNameCheckVisitorBase):
    @assert_passes()
    def test_asyncio_coroutine(self):
        import asyncio
        from collections.abc import Awaitable

        @asyncio.coroutine
        def f():
            yield from asyncio.sleep(3)
            return 42

        @asyncio.coroutine
        def g():
            assert_is_value(f(), GenericValue(Awaitable, [KnownValue(42)]))

    @assert_passes()
    def test_coroutine_from_typeshed(self):
        import asyncio

        async def capybara():
            # annotated as def ... -> Future in typeshed
            assert_is_value(
                asyncio.sleep(3), GenericValue(asyncio.Future, [KnownValue(None)])
            )
            return 42

    @assert_passes()
    def test_async_def_from_typeshed(self):
        from asyncio.streams import open_connection, StreamReader, StreamWriter
        from collections.abc import Awaitable

        async def capybara():
            # annotated as async def in typeshed
            assert_is_value(
                open_connection(),
                GenericValue(
                    Awaitable,
                    [
                        SequenceIncompleteValue(
                            tuple, [TypedValue(StreamReader), TypedValue(StreamWriter)]
                        )
                    ],
                ),
            )
            return 42

    @assert_passes()
    def test_async_def(self):
        from collections.abc import Awaitable

        async def f():
            return 42

        async def g():
            assert_is_value(f(), GenericValue(Awaitable, [KnownValue(42)]))


class TestTypeshedClient(TestNameCheckVisitorBase):
    @assert_passes()
    def test_types(self):
        import math

        assert_is_value(math.exp(1.0), TypedValue(float))
        assert_is_value("".isspace(), TypedValue(bool))

    @assert_passes()
    def test_dict_update(self):
        def capybara():
            x = {}
            x.update({})  # just check that this doesn't fail

    def test_get_bases(self):
        tsf = TypeshedFinder(verbose=True)
        assert_eq(
            [
                GenericValue(MutableSequence, (TypeVarValue(typevar=Anything),)),
                GenericValue(Generic, (TypeVarValue(typevar=Anything),)),
            ],
            tsf.get_bases(list),
        )
        assert_eq(
            [
                GenericValue(Collection, (TypeVarValue(typevar=Anything),)),
                GenericValue(Reversible, (TypeVarValue(typevar=Anything),)),
                GenericValue(Generic, (TypeVarValue(typevar=Anything),)),
            ],
            tsf.get_bases(Sequence),
        )
        assert_eq(
            [GenericValue(Collection, (TypeVarValue(Anything),))], tsf.get_bases(Set)
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

    @assert_fails(ErrorCode.incompatible_argument)
    def test_only_T(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Capybara(Generic[T]):
            def add_one(self, obj: T) -> None:
                pass

        def capybara(x: Capybara[int]) -> None:
            x.add_one("x")

    @assert_passes()
    def test_multi_typevar(self):
        from typing import TypeVar, Optional

        T = TypeVar("T")

        # inspired by tempfile.mktemp
        def mktemp(prefix: Optional[T] = None, suffix: Optional[T] = None) -> T:
            raise NotImplementedError

        def capybara() -> None:
            assert_is_value(mktemp(), UNRESOLVED_VALUE)
            assert_is_value(mktemp(prefix="p"), KnownValue("p"))
            assert_is_value(mktemp(suffix="s"), KnownValue("s"))

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
    @assert_fails(ErrorCode.incompatible_argument)
    def test_typeshed(self):
        from typing import List

        def capybara(lst: List[int]) -> None:
            lst.append("x")

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


class Parent(Generic[T]):
    pass


class Child(Parent[int]):
    pass


class GenericChild(Parent[T]):
    pass


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


class TestGetGenericBases:
    def setup(self) -> None:
        arg_spec_cache = ArgSpecCache(TestConfig())
        self.get_generic_bases = arg_spec_cache.get_generic_bases

    def test_runtime(self):
        assert_eq({Parent: [UNRESOLVED_VALUE]}, self.get_generic_bases(Parent))
        assert_eq(
            {Parent: [TypeVarValue(T)]},
            self.get_generic_bases(Parent, [TypeVarValue(T)]),
        )
        assert_eq({Child: [], Parent: [TypedValue(int)]}, self.get_generic_bases(Child))
        assert_eq(
            {GenericChild: [UNRESOLVED_VALUE], Parent: [UNRESOLVED_VALUE]},
            self.get_generic_bases(GenericChild),
        )
        one = KnownValue(1)
        assert_eq(
            {GenericChild: [one], Parent: [one]},
            self.get_generic_bases(GenericChild, [one]),
        )

    def test_callable(self):
        assert_eq(
            {collections.abc.Callable: [], object: []},
            self.get_generic_bases(collections.abc.Callable, []),
        )

    def test_struct_time(self):
        assert_eq(
            {
                time.struct_time: [],
                # Ideally should be not Any, but we haven't implemented
                # support for typeshed namedtuples.
                tuple: [UNRESOLVED_VALUE],
                collections.abc.Collection: [UNRESOLVED_VALUE],
                collections.abc.Reversible: [UNRESOLVED_VALUE],
                collections.abc.Iterable: [UNRESOLVED_VALUE],
                collections.abc.Sequence: [UNRESOLVED_VALUE],
                collections.abc.Container: [UNRESOLVED_VALUE],
            },
            self.get_generic_bases(time.struct_time, []),
        )

    def test_context_manager(self):
        int_tv = TypedValue(int)
        assert_eq(
            {contextlib.AbstractContextManager: [int_tv]},
            self.get_generic_bases(contextlib.AbstractContextManager, [int_tv]),
        )

    def test_collections(self):
        int_tv = TypedValue(int)
        str_tv = TypedValue(str)
        assert_eq(
            {
                collections.abc.ValuesView: [int_tv],
                collections.abc.MappingView: [],
                collections.abc.Iterable: [int_tv],
                collections.abc.Sized: [],
            },
            self.get_generic_bases(collections.abc.ValuesView, [int_tv]),
        )
        assert_eq(
            {
                collections.deque: [int_tv],
                collections.abc.MutableSequence: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Reversible: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Sequence: [int_tv],
                collections.abc.Container: [int_tv],
            },
            self.get_generic_bases(collections.deque, [int_tv]),
        )
        assert_eq(
            {
                collections.defaultdict: [int_tv, str_tv],
                dict: [int_tv, str_tv],
                collections.abc.MutableMapping: [int_tv, str_tv],
                collections.abc.Mapping: [int_tv, str_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [int_tv],
            },
            self.get_generic_bases(collections.defaultdict, [int_tv, str_tv]),
        )

    def test_typeshed(self):
        int_tv = TypedValue(int)
        str_tv = TypedValue(str)
        assert_eq(
            {
                list: [int_tv],
                collections.abc.MutableSequence: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Reversible: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Sequence: [int_tv],
                collections.abc.Container: [int_tv],
            },
            self.get_generic_bases(list, [int_tv]),
        )
        assert_eq(
            {
                set: [int_tv],
                collections.abc.MutableSet: [int_tv],
                collections.abc.Set: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [int_tv],
            },
            self.get_generic_bases(set, [int_tv]),
        )
        assert_eq(
            {
                dict: [int_tv, str_tv],
                collections.abc.MutableMapping: [int_tv, str_tv],
                collections.abc.Mapping: [int_tv, str_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [int_tv],
            },
            self.get_generic_bases(dict, [int_tv, str_tv]),
        )
