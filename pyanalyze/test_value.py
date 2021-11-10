import collections.abc
import enum
import io
import pickle
from qcore.asserts import (
    assert_eq,
    assert_in,
    assert_is,
    assert_is_instance,
    assert_is_not,
)
from typing import NewType, Sequence, Dict
from typing_extensions import Protocol, runtime_checkable
import typing
import types
from unittest import mock


from . import tests
from . import value
from .arg_spec import ArgSpecCache
from .stacked_scopes import Composite
from .test_config import TestConfig
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CanAssignError,
    Value,
    GenericValue,
    KnownValue,
    TypedValue,
    MultiValuedValue,
    SubclassValue,
    CanAssignContext,
    SequenceIncompleteValue,
    TypeVarMap,
    concrete_values_from_iterable,
)


class Context(CanAssignContext):
    def __init__(self) -> None:
        self.arg_spec_cache = ArgSpecCache(TestConfig())

    def get_generic_bases(
        self, typ: type, generic_args: Sequence[Value] = ()
    ) -> Dict[type, TypeVarMap]:
        return self.arg_spec_cache.get_generic_bases(typ, generic_args)


CTX = Context()


def assert_cannot_assign(left: Value, right: Value) -> None:
    tv_map = left.can_assign(right, CTX)
    assert_is_instance(tv_map, CanAssignError)
    print(tv_map.display())


def assert_can_assign(left: Value, right: Value, typevar_map: TypeVarMap = {}) -> None:
    assert_eq(typevar_map, left.can_assign(right, CTX))


def test_any_value() -> None:
    any = AnyValue(AnySource.unannotated)
    assert not any.is_type(int)
    assert_can_assign(any, KnownValue(1))
    assert_can_assign(any, TypedValue(int))
    assert_can_assign(any, MultiValuedValue([KnownValue(1), TypedValue(int)]))


def test_known_value() -> None:
    val = KnownValue(3)
    assert_eq(3, val.val)
    assert_eq("Literal[3]", str(val))
    assert_eq("Literal['']", str(KnownValue("")))
    assert val.is_type(int)
    assert not val.is_type(str)

    assert_cannot_assign(val, KnownValue(1))
    assert_can_assign(val, val)
    assert_cannot_assign(val, TypedValue(int))
    assert_can_assign(val, MultiValuedValue([val, AnyValue(AnySource.marker)]))
    assert_cannot_assign(val, val | TypedValue(int))
    assert_cannot_assign(KnownValue(int), SubclassValue(TypedValue(int)))
    assert_cannot_assign(KnownValue(1), KnownValue(True))
    assert_cannot_assign(KnownValue(True), KnownValue(1))

    nan = float("nan")
    assert_can_assign(KnownValue(nan), KnownValue(nan))
    assert_cannot_assign(KnownValue(nan), KnownValue(0.0))


def test_unbound_method_value() -> None:
    val = value.UnboundMethodValue(
        "get_prop_with_get", Composite(value.TypedValue(tests.PropertyObject))
    )
    assert_eq("<method get_prop_with_get on pyanalyze.tests.PropertyObject>", str(val))
    assert_eq("get_prop_with_get", val.attr_name)
    assert_eq(TypedValue(tests.PropertyObject), val.composite.value)
    assert_is(None, val.secondary_attr_name)
    assert_eq(tests.PropertyObject.get_prop_with_get, val.get_method())
    assert val.is_type(object)
    assert not val.is_type(str)

    val = value.UnboundMethodValue(
        "get_prop_with_get",
        Composite(value.TypedValue(tests.PropertyObject)),
        secondary_attr_name="asynq",
    )
    assert_eq(
        "<method get_prop_with_get.asynq on pyanalyze.tests.PropertyObject>", str(val)
    )
    assert_eq("get_prop_with_get", val.attr_name)
    assert_eq(TypedValue(tests.PropertyObject), val.composite.value)
    assert_eq("asynq", val.secondary_attr_name)
    method = val.get_method()
    assert_is_not(None, method)
    assert_in(method.__name__, tests.ASYNQ_METHOD_NAMES)
    assert_eq(tests.PropertyObject.get_prop_with_get, method.__self__)
    assert val.is_type(object)
    assert not val.is_type(str)


def test_typed_value() -> None:
    val = TypedValue(str)
    assert_is(str, val.typ)
    assert_eq("str", str(val))
    assert val.is_type(str)
    assert not val.is_type(int)

    assert_can_assign(val, val)
    assert_cannot_assign(val, TypedValue(int))
    assert_can_assign(val, KnownValue("x"))
    assert_can_assign(val, MultiValuedValue([val, KnownValue("x")]))
    assert_cannot_assign(val, MultiValuedValue([KnownValue("x"), TypedValue(int)]))

    float_val = TypedValue(float)
    assert_eq("float", str(float_val))
    assert_can_assign(float_val, KnownValue(1.0))
    assert_can_assign(float_val, KnownValue(1))
    assert_cannot_assign(float_val, KnownValue(""))
    assert_can_assign(float_val, TypedValue(float))
    assert_can_assign(float_val, TypedValue(int))
    assert_cannot_assign(float_val, TypedValue(str))
    assert_can_assign(float_val, TypedValue(mock.Mock))

    assert_cannot_assign(float_val, SubclassValue(TypedValue(float)))
    assert_can_assign(TypedValue(type), SubclassValue(TypedValue(float)))


@runtime_checkable
class Proto(Protocol):
    def asynq(self) -> None:
        ...


def test_protocol() -> None:
    tv = TypedValue(Proto)

    def fn() -> None:
        pass

    assert_cannot_assign(tv, KnownValue(fn))
    fn.asynq = lambda: None
    assert_can_assign(tv, KnownValue(fn))

    class X:
        def asynq(self) -> None:
            pass

    assert_can_assign(tv, TypedValue(X))
    assert_can_assign(tv, KnownValue(X()))


def test_callable() -> None:
    cval = TypedValue(collections.abc.Callable)
    assert_can_assign(cval, cval)

    gen_val = GenericValue(
        collections.abc.Callable, [TypedValue(int), KnownValue(None)]
    )
    assert_can_assign(gen_val, gen_val)


def test_subclass_value() -> None:
    val = SubclassValue(TypedValue(int))
    assert_can_assign(val, KnownValue(int))
    assert_can_assign(val, KnownValue(bool))
    assert_cannot_assign(val, KnownValue(str))
    assert_can_assign(val, TypedValue(type))
    assert_cannot_assign(val, TypedValue(int))
    assert_can_assign(val, SubclassValue(TypedValue(bool)))
    assert_can_assign(val, TypedValue(type))
    assert_cannot_assign(val, SubclassValue(TypedValue(str)))
    val = SubclassValue(TypedValue(str))
    assert_eq("Type[str]", str(val))
    assert_eq(TypedValue(str), val.typ)
    assert val.is_type(str)
    assert not val.is_type(int)
    val = SubclassValue(TypedValue(float))
    assert_can_assign(val, KnownValue(int))
    assert_can_assign(val, SubclassValue(TypedValue(int)))


def test_generic_value() -> None:
    val = GenericValue(list, [TypedValue(int)])
    assert_eq("list[int]", str(val))
    assert_can_assign(val, TypedValue(list))
    assert_can_assign(val, GenericValue(list, [AnyValue(AnySource.marker)]))
    assert_can_assign(val, GenericValue(list, [TypedValue(bool)]))
    assert_cannot_assign(val, GenericValue(list, [TypedValue(str)]))
    assert_cannot_assign(val, GenericValue(set, [TypedValue(int)]))
    assert_eq("tuple[int, ...]", str(value.GenericValue(tuple, [TypedValue(int)])))

    it = GenericValue(collections.abc.Iterable, [TypedValue(object)])
    assert_can_assign(
        it, GenericValue(types.GeneratorType, [TypedValue(bool) | KnownValue(None)])
    )


def test_sequence_incomplete_value() -> None:
    val = value.SequenceIncompleteValue(tuple, [TypedValue(int), TypedValue(str)])
    assert_can_assign(val, TypedValue(tuple))
    assert_can_assign(val, GenericValue(tuple, [TypedValue(int) | TypedValue(str)]))
    assert_cannot_assign(val, GenericValue(tuple, [TypedValue(int) | TypedValue(list)]))

    assert_can_assign(val, val)
    assert_cannot_assign(val, value.SequenceIncompleteValue(tuple, [TypedValue(int)]))
    assert_can_assign(
        val, value.SequenceIncompleteValue(tuple, [TypedValue(bool), TypedValue(str)])
    )

    assert_eq("tuple[int, str]", str(val))
    assert_eq(
        "tuple[int]", str(value.SequenceIncompleteValue(tuple, [TypedValue(int)]))
    )
    assert_eq(
        "<list containing [int]>",
        str(value.SequenceIncompleteValue(list, [TypedValue(int)])),
    )


def test_dict_incomplete_value() -> None:
    val = value.DictIncompleteValue(dict, [(TypedValue(int), KnownValue("x"))])
    assert_eq("<dict containing {int: Literal['x']}>", str(val))


def test_multi_valued_value() -> None:
    val = TypedValue(int) | KnownValue(None)
    assert_eq(MultiValuedValue([TypedValue(int), KnownValue(None)]), val)
    assert_eq(val, val | KnownValue(None))
    assert_eq(
        MultiValuedValue([TypedValue(int), KnownValue(None), TypedValue(str)]),
        val | TypedValue(str),
    )

    assert_eq("Optional[int]", str(val))
    assert_can_assign(val, KnownValue(1))
    assert_can_assign(val, KnownValue(None))
    assert_cannot_assign(val, KnownValue(""))
    assert_cannot_assign(val, TypedValue(float))
    assert_can_assign(val, val)
    assert_cannot_assign(val, KnownValue(None) | TypedValue(str))
    assert_can_assign(
        val, AnyValue(AnySource.marker) | TypedValue(int) | KnownValue(None)
    )

    assert_eq("Literal[1, 2]", str(KnownValue(1) | KnownValue(2)))
    assert_eq(
        "Literal[1, 2, None]", str(KnownValue(1) | KnownValue(2) | KnownValue(None))
    )
    assert_eq("Union[int, str]", str(TypedValue(int) | TypedValue(str)))
    assert_eq(
        "Union[int, str, None]",
        str(TypedValue(int) | TypedValue(str) | KnownValue(None)),
    )
    assert_eq(
        "Union[int, str, Literal[1, 2], None]",
        str(
            TypedValue(int)
            | TypedValue(str)
            | KnownValue(None)
            | KnownValue(1)
            | KnownValue(2)
        ),
    )


def test_large_union_optimization() -> None:
    val = MultiValuedValue([*[KnownValue(i) for i in range(10000)], TypedValue(str)])
    assert_can_assign(val, KnownValue(1))
    assert_cannot_assign(val, KnownValue(234234))
    assert_cannot_assign(val, KnownValue(True))
    assert_can_assign(val, KnownValue(""))


class ThriftEnum(object):
    X = 0
    Y = 1

    _VALUES_TO_NAMES = {0: "X", 1: "Y"}

    _NAMES_TO_VALUES = {"X": 0, "Y": 1}


def test_can_assign_thrift_enum() -> None:
    val = TypedValue(ThriftEnum)
    assert_can_assign(val, KnownValue(0))
    assert_cannot_assign(val, KnownValue(2))
    assert_cannot_assign(val, KnownValue(1.0))

    assert_can_assign(val, TypedValue(int))
    assert_can_assign(val, TypedValue(ThriftEnum))
    assert_cannot_assign(val, TypedValue(str))


def test_variable_name_value() -> None:
    uid_val = value.VariableNameValue(["uid", "viewer"])
    varname_map = {
        "uid": uid_val,
        "viewer": uid_val,
        "actor_id": value.VariableNameValue(["actor_id"]),
    }

    assert_is(None, value.VariableNameValue.from_varname("capybaras", varname_map))

    val = value.VariableNameValue.from_varname("uid", varname_map)
    assert_is_not(None, val)
    assert_is(val, value.VariableNameValue.from_varname("viewer", varname_map))
    assert_is(val, value.VariableNameValue.from_varname("old_uid", varname_map))
    assert_is_not(val, value.VariableNameValue.from_varname("actor_id", varname_map))
    assert_can_assign(TypedValue(int), val)
    assert_can_assign(KnownValue(1), val)
    assert_can_assign(val, TypedValue(int))
    assert_can_assign(val, KnownValue(1))


def test_typeddict_value() -> None:
    val = value.TypedDictValue(
        {"a": (True, TypedValue(int)), "b": (True, TypedValue(str))}
    )
    # dict iteration order in some Python versions is not deterministic
    assert_in(
        str(val), ['TypedDict({"a": int, "b": str})', 'TypedDict({"b": str, "a": int})']
    )

    assert_can_assign(val, AnyValue(AnySource.marker))
    assert_can_assign(val, TypedValue(dict))
    assert_cannot_assign(val, TypedValue(str))

    # KnownValue of dict
    assert_can_assign(val, KnownValue({"a": 1, "b": "2"}))
    # extra keys are ok
    assert_can_assign(val, KnownValue({"a": 1, "b": "2", "c": 1}))
    # missing key
    assert_cannot_assign(val, KnownValue({"a": 1}))
    # wrong type
    assert_cannot_assign(val, KnownValue({"a": 1, "b": 2}))

    # TypedDictValue
    assert_can_assign(val, val)
    assert_can_assign(
        val,
        value.TypedDictValue(
            {"a": (True, KnownValue(1)), "b": (True, TypedValue(str))}
        ),
    )
    assert_can_assign(
        val,
        value.TypedDictValue(
            {
                "a": (True, KnownValue(1)),
                "b": (True, TypedValue(str)),
                "c": (True, TypedValue(float)),
            }
        ),
    )
    assert_cannot_assign(
        val,
        value.TypedDictValue(
            {"a": (True, KnownValue(1)), "b": (True, TypedValue(int))}
        ),
    )
    assert_cannot_assign(val, value.TypedDictValue({"b": (True, TypedValue(str))}))

    # DictIncompleteValue
    assert_can_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [(KnownValue("a"), TypedValue(int)), (KnownValue("b"), TypedValue(str))],
        ),
    )
    assert_can_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                (KnownValue("a"), TypedValue(int)),
                (KnownValue("b"), TypedValue(str)),
                (KnownValue("c"), AnyValue(AnySource.marker)),
            ],
        ),
    )
    assert_can_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                (KnownValue("a"), TypedValue(int)),
                (AnyValue(AnySource.marker), TypedValue(str)),
            ],
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict, [(AnyValue(AnySource.marker), TypedValue(str))]
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [(KnownValue("a"), TypedValue(int)), (KnownValue("b"), TypedValue(float))],
        ),
    )


class Capybara(enum.IntEnum):
    hydrochaeris = 1
    isthmius = 2


def test_new_type_value() -> None:
    nt1 = NewType("nt1", int)
    nt1_val = value.NewTypeValue(nt1)
    nt2 = NewType("nt2", int)
    nt2_val = value.NewTypeValue(nt2)
    assert_can_assign(nt1_val, nt1_val)
    assert_cannot_assign(nt1_val, nt2_val)
    # This should eventually return False
    assert_can_assign(nt1_val, TypedValue(int))
    assert_can_assign(TypedValue(int), nt1_val)
    assert_cannot_assign(nt1_val, TypedValue(Capybara))
    assert_cannot_assign(nt1_val, KnownValue(Capybara.hydrochaeris))


def test_annotated_value() -> None:
    tv_int = TypedValue(int)
    assert_can_assign(AnnotatedValue(tv_int, [tv_int]), tv_int)
    assert_can_assign(tv_int, AnnotatedValue(tv_int, [tv_int]))


def test_io() -> None:
    assert_can_assign(
        GenericValue(typing.IO, [AnyValue(AnySource.marker)]), TypedValue(io.BytesIO)
    )


def test_concrete_values_from_iterable() -> None:
    assert_is(None, concrete_values_from_iterable(KnownValue(1), CTX))
    assert_eq((), concrete_values_from_iterable(KnownValue(()), CTX))
    assert_eq(
        (KnownValue(1), KnownValue(2)),
        concrete_values_from_iterable(KnownValue((1, 2)), CTX),
    )
    assert_eq(
        MultiValuedValue((KnownValue(1), KnownValue(2))),
        concrete_values_from_iterable(
            SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]), CTX
        ),
    )
    assert_eq(
        TypedValue(int),
        concrete_values_from_iterable(GenericValue(list, [TypedValue(int)]), CTX),
    )
    assert_eq(
        MultiValuedValue([KnownValue(1), KnownValue(3), KnownValue(2), KnownValue(4)]),
        concrete_values_from_iterable(
            MultiValuedValue(
                [
                    SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]),
                    KnownValue((3, 4)),
                ]
            ),
            CTX,
        ),
    )
    assert_eq(
        MultiValuedValue([KnownValue(1), KnownValue(2), TypedValue(int)]),
        concrete_values_from_iterable(
            MultiValuedValue(
                [
                    SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]),
                    GenericValue(list, [TypedValue(int)]),
                ]
            ),
            CTX,
        ),
    )
    assert_eq(
        MultiValuedValue([KnownValue(1), KnownValue(2), KnownValue(3)]),
        concrete_values_from_iterable(
            MultiValuedValue(
                [
                    SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]),
                    KnownValue((3,)),
                ]
            ),
            CTX,
        ),
    )


def _assert_pickling_roundtrip(obj: object) -> None:
    assert_eq(obj, pickle.loads(pickle.dumps(obj)))


def test_pickling() -> None:
    _assert_pickling_roundtrip(KnownValue(1))
    _assert_pickling_roundtrip(TypedValue(int))
    _assert_pickling_roundtrip(KnownValue(None) | TypedValue(str))
