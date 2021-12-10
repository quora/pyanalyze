import ast
import collections.abc
import enum
import io
import pickle
from typing import NewType
from typing_extensions import Protocol, runtime_checkable
import typing
import types
from unittest import mock


from . import tests
from . import value
from .checker import Checker
from .name_check_visitor import NameCheckVisitor
from .signature import Signature
from .stacked_scopes import Composite
from .test_config import TestConfig
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignError,
    KVPair,
    Value,
    GenericValue,
    KnownValue,
    TypedValue,
    MultiValuedValue,
    SubclassValue,
    SequenceIncompleteValue,
    TypeVarMap,
    concrete_values_from_iterable,
)

_checker = Checker(TestConfig())
CTX = NameCheckVisitor("", "", ast.parse(""), checker=_checker)


def assert_cannot_assign(left: Value, right: Value) -> None:
    tv_map = left.can_assign(right, CTX)
    assert isinstance(tv_map, CanAssignError)
    print(tv_map.display())


def assert_can_assign(left: Value, right: Value, typevar_map: TypeVarMap = {}) -> None:
    assert typevar_map == left.can_assign(right, CTX)


def test_any_value() -> None:
    any = AnyValue(AnySource.unannotated)
    assert not any.is_type(int)
    assert_can_assign(any, KnownValue(1))
    assert_can_assign(any, TypedValue(int))
    assert_can_assign(any, MultiValuedValue([KnownValue(1), TypedValue(int)]))


def test_known_value() -> None:
    val = KnownValue(3)
    assert 3 == val.val
    assert "Literal[3]" == str(val)
    assert "Literal['']" == str(KnownValue(""))
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
    po_composite = Composite(value.TypedValue(tests.PropertyObject))
    val = value.UnboundMethodValue("get_prop_with_get", po_composite)
    assert "<method get_prop_with_get on pyanalyze.tests.PropertyObject>" == str(val)
    assert "get_prop_with_get" == val.attr_name
    assert TypedValue(tests.PropertyObject) == val.composite.value
    assert None is val.secondary_attr_name
    assert tests.PropertyObject.get_prop_with_get == val.get_method()
    assert val.is_type(object)
    assert not val.is_type(str)

    val = value.UnboundMethodValue(
        "get_prop_with_get", po_composite, secondary_attr_name="asynq"
    )
    assert "<method get_prop_with_get.asynq on pyanalyze.tests.PropertyObject>" == str(
        val
    )
    assert "get_prop_with_get" == val.attr_name
    assert TypedValue(tests.PropertyObject) == val.composite.value
    assert "asynq" == val.secondary_attr_name
    method = val.get_method()
    assert method is not None
    assert method.__name__ in tests.ASYNQ_METHOD_NAMES
    assert tests.PropertyObject.get_prop_with_get == method.__self__
    assert val.is_type(object)
    assert not val.is_type(str)

    val = value.UnboundMethodValue("non_async_method", po_composite)
    assert val.get_method() is not None
    assert val.get_signature(CTX) is not None
    assert_can_assign(val, val)
    assert_cannot_assign(val, KnownValue(1))
    assert_can_assign(val, CallableValue(Signature.make([], is_ellipsis_args=True)))
    assert_can_assign(val, CallableValue(Signature.make([])))


def test_typed_value() -> None:
    val = TypedValue(str)
    assert str is val.typ
    assert "str" == str(val)
    assert val.is_type(str)
    assert not val.is_type(int)

    assert_can_assign(val, val)
    assert_cannot_assign(val, TypedValue(int))
    assert_can_assign(val, KnownValue("x"))
    assert_can_assign(val, MultiValuedValue([val, KnownValue("x")]))
    assert_cannot_assign(val, MultiValuedValue([KnownValue("x"), TypedValue(int)]))

    float_val = TypedValue(float)
    assert "float" == str(float_val)
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
    assert "Type[str]" == str(val)
    assert TypedValue(str) == val.typ
    assert val.is_type(str)
    assert not val.is_type(int)
    val = SubclassValue(TypedValue(float))
    assert_can_assign(val, KnownValue(int))
    assert_can_assign(val, SubclassValue(TypedValue(int)))


def test_generic_value() -> None:
    val = GenericValue(list, [TypedValue(int)])
    assert "list[int]" == str(val)
    assert_can_assign(val, TypedValue(list))
    assert_can_assign(val, GenericValue(list, [AnyValue(AnySource.marker)]))
    assert_can_assign(val, GenericValue(list, [TypedValue(bool)]))
    assert_cannot_assign(val, GenericValue(list, [TypedValue(str)]))
    assert_cannot_assign(val, GenericValue(set, [TypedValue(int)]))
    assert "tuple[int, ...]" == str(value.GenericValue(tuple, [TypedValue(int)]))

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

    assert "tuple[int, str]" == str(val)
    assert "tuple[int]" == str(value.SequenceIncompleteValue(tuple, [TypedValue(int)]))
    assert "<list containing [int]>" == str(
        value.SequenceIncompleteValue(list, [TypedValue(int)])
    )


def test_dict_incomplete_value() -> None:
    val = value.DictIncompleteValue(dict, [KVPair(TypedValue(int), KnownValue("x"))])
    assert "<dict containing {int: Literal['x']}>" == str(val)

    val = value.DictIncompleteValue(
        dict,
        [
            KVPair(KnownValue("a"), TypedValue(int)),
            KVPair(KnownValue("b"), TypedValue(str)),
        ],
    )
    assert val.get_value(KnownValue("a"), CTX) == TypedValue(int)


def test_multi_valued_value() -> None:
    val = TypedValue(int) | KnownValue(None)
    assert MultiValuedValue([TypedValue(int), KnownValue(None)]) == val
    assert val == val | KnownValue(None)
    assert MultiValuedValue(
        [TypedValue(int), KnownValue(None), TypedValue(str)]
    ) == val | TypedValue(str)

    assert "Optional[int]" == str(val)
    assert_can_assign(val, KnownValue(1))
    assert_can_assign(val, KnownValue(None))
    assert_cannot_assign(val, KnownValue(""))
    assert_cannot_assign(val, TypedValue(float))
    assert_can_assign(val, val)
    assert_cannot_assign(val, KnownValue(None) | TypedValue(str))
    assert_can_assign(
        val, AnyValue(AnySource.marker) | TypedValue(int) | KnownValue(None)
    )

    assert "Literal[1, 2]" == str(KnownValue(1) | KnownValue(2))
    assert "Literal[1, 2, None]" == str(
        KnownValue(1) | KnownValue(2) | KnownValue(None)
    )
    assert "Union[int, str]" == str(TypedValue(int) | TypedValue(str))
    assert "Union[int, str, None]" == str(
        TypedValue(int) | TypedValue(str) | KnownValue(None)
    )
    assert "Union[int, str, Literal[1, 2], None]" == str(
        TypedValue(int)
        | TypedValue(str)
        | KnownValue(None)
        | KnownValue(1)
        | KnownValue(2)
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

    assert None is value.VariableNameValue.from_varname("capybaras", varname_map)

    val = value.VariableNameValue.from_varname("uid", varname_map)
    assert None is not val
    assert val is value.VariableNameValue.from_varname("viewer", varname_map)
    assert val is value.VariableNameValue.from_varname("old_uid", varname_map)
    assert val is not value.VariableNameValue.from_varname("actor_id", varname_map)
    assert_can_assign(TypedValue(int), val)
    assert_can_assign(KnownValue(1), val)
    assert_can_assign(val, TypedValue(int))
    assert_can_assign(val, KnownValue(1))


def test_typeddict_value() -> None:
    val = value.TypedDictValue(
        {"a": (True, TypedValue(int)), "b": (True, TypedValue(str))}
    )
    # dict iteration order in some Python versions is not deterministic
    assert str(val) in [
        'TypedDict({"a": int, "b": str})',
        'TypedDict({"b": str, "a": int})',
    ]

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
            [
                KVPair(KnownValue("a"), TypedValue(int)),
                KVPair(KnownValue("b"), TypedValue(str)),
            ],
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(str)),
                KVPair(KnownValue("b"), TypedValue(str)),
            ],
        ),
    )
    assert_can_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(str)),
                KVPair(KnownValue("a"), TypedValue(int)),
                KVPair(KnownValue("b"), TypedValue(str)),
            ],
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(str)),
                KVPair(KnownValue("a"), TypedValue(int), is_required=False),
                KVPair(KnownValue("b"), TypedValue(str)),
            ],
        ),
    )
    assert_can_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(int)),
                KVPair(KnownValue("b"), TypedValue(str)),
                KVPair(KnownValue("c"), AnyValue(AnySource.marker)),
            ],
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(int)),
                KVPair(AnyValue(AnySource.marker), TypedValue(str)),
            ],
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict, [KVPair(AnyValue(AnySource.marker), TypedValue(str))]
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(int)),
                KVPair(KnownValue("b"), TypedValue(float)),
            ],
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
    assert isinstance(concrete_values_from_iterable(KnownValue(1), CTX), CanAssignError)
    assert () == concrete_values_from_iterable(KnownValue(()), CTX)
    assert (KnownValue(1), KnownValue(2)) == concrete_values_from_iterable(
        KnownValue((1, 2)), CTX
    )
    assert (KnownValue(1), KnownValue(2)) == concrete_values_from_iterable(
        SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]), CTX
    )
    assert TypedValue(int) == concrete_values_from_iterable(
        GenericValue(list, [TypedValue(int)]), CTX
    )
    assert [
        KnownValue(1) | KnownValue(3),
        KnownValue(2) | KnownValue(4),
    ] == concrete_values_from_iterable(
        MultiValuedValue(
            [
                SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]),
                KnownValue((3, 4)),
            ]
        ),
        CTX,
    )
    assert MultiValuedValue(
        [KnownValue(1), KnownValue(2), TypedValue(int)]
    ) == concrete_values_from_iterable(
        MultiValuedValue(
            [
                SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]),
                GenericValue(list, [TypedValue(int)]),
            ]
        ),
        CTX,
    )
    assert MultiValuedValue(
        [KnownValue(1), KnownValue(2), KnownValue(3)]
    ) == concrete_values_from_iterable(
        MultiValuedValue(
            [
                SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]),
                KnownValue((3,)),
            ]
        ),
        CTX,
    )

    class HasGetItem:
        def __getitem__(self, i: int) -> str:
            return str(i)

    assert concrete_values_from_iterable(TypedValue(HasGetItem), CTX) == TypedValue(str)

    class BadGetItem:
        def __getitem__(self, i: int, extra: bool) -> str:
            return str(i) + str(extra)

    assert isinstance(
        concrete_values_from_iterable(TypedValue(BadGetItem), CTX), CanAssignError
    )


def _assert_pickling_roundtrip(obj: object) -> None:
    assert obj == pickle.loads(pickle.dumps(obj))


def test_pickling() -> None:
    _assert_pickling_roundtrip(KnownValue(1))
    _assert_pickling_roundtrip(TypedValue(int))
    _assert_pickling_roundtrip(KnownValue(None) | TypedValue(str))
