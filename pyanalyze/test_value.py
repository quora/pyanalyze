from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from qcore.asserts import assert_eq, assert_in, assert_is, assert_is_not

from . import tests
from . import value
from .value import KnownValue, TypedValue, MultiValuedValue, SubclassValue
from .test_node_visitor import skip_before


def test_UNRESOLVED_VALUE():
    assert not value.UNRESOLVED_VALUE.is_type(int)


def test_known_value():
    val = KnownValue(3)
    assert_eq(3, val.val)
    assert_eq("Literal[3]", str(val))
    assert_eq("Literal['']", str(KnownValue("")))
    assert val.is_type(int)
    assert not val.is_type(str)
    assert val.is_value_compatible(KnownValue(3))
    assert val.is_value_compatible(TypedValue(int))
    assert val.is_value_compatible(MultiValuedValue([KnownValue(3), TypedValue(int)]))
    assert not val.is_value_compatible(
        MultiValuedValue([KnownValue("x"), TypedValue(int)])
    )

    assert TypedValue(int).is_value_compatible(val)
    assert not TypedValue(str).is_value_compatible(val)

    assert not val.is_value_compatible(SubclassValue(int))
    assert KnownValue(int).is_value_compatible(SubclassValue(int))
    assert not KnownValue(str).is_value_compatible(SubclassValue(int))


def test_unbound_method_value():
    val = value.UnboundMethodValue("get_prop_with_get", tests.PropertyObject)
    assert_eq("<method get_prop_with_get on pyanalyze.tests.PropertyObject>", str(val))
    assert_eq("get_prop_with_get", val.attr_name)
    assert_is(tests.PropertyObject, val.typ)
    assert_is(None, val.secondary_attr_name)
    assert_eq(tests.PropertyObject.get_prop_with_get, val.get_method())
    assert val.is_type(object)
    assert not val.is_type(str)

    val = value.UnboundMethodValue(
        "get_prop_with_get", tests.PropertyObject, secondary_attr_name="asynq"
    )
    assert_eq(
        "<method get_prop_with_get.asynq on pyanalyze.tests.PropertyObject>", str(val)
    )
    assert_eq("get_prop_with_get", val.attr_name)
    assert_is(tests.PropertyObject, val.typ)
    assert_eq("asynq", val.secondary_attr_name)
    method = val.get_method()
    assert_in(method.__name__, tests.ASYNQ_METHOD_NAMES)
    assert_eq(tests.PropertyObject.get_prop_with_get, method.__self__)
    assert val.is_type(object)
    assert not val.is_type(str)


def test_typed_value():
    val = TypedValue(str)
    assert_is(str, val.typ)
    assert_eq("str", str(val))
    assert val.is_type(str)
    assert not val.is_type(int)

    assert val.is_value_compatible(TypedValue(str))
    assert not val.is_value_compatible(TypedValue(int))
    assert val.is_value_compatible(MultiValuedValue([KnownValue("x"), TypedValue(str)]))
    assert not val.is_value_compatible(
        MultiValuedValue([KnownValue("x"), TypedValue(int)])
    )

    float_val = TypedValue(float)
    assert_eq("float", str(float_val))
    assert float_val.is_value_compatible(KnownValue(1.0))
    assert float_val.is_value_compatible(KnownValue(1))
    assert not float_val.is_value_compatible(KnownValue(""))
    assert float_val.is_value_compatible(TypedValue(float))
    assert float_val.is_value_compatible(TypedValue(int))
    assert not float_val.is_value_compatible(TypedValue(str))
    assert float_val.is_value_compatible(TypedValue(value.mock.Mock))

    assert not float_val.is_value_compatible(SubclassValue(float))
    assert TypedValue(type).is_value_compatible(SubclassValue(float))


def test_subclass_value():
    val = SubclassValue(int)
    assert val.is_value_compatible(KnownValue(int))
    assert val.is_value_compatible(KnownValue(bool))
    assert not val.is_value_compatible(KnownValue(str))
    assert val.is_value_compatible(TypedValue(type))
    assert not val.is_value_compatible(TypedValue(int))
    assert val.is_value_compatible(SubclassValue(bool))
    assert not val.is_value_compatible(SubclassValue(str))


def test_generic_value():
    val = value.GenericValue(list, [TypedValue(int)])
    assert_eq("list[int]", str(val))
    assert val.is_value_compatible(TypedValue(list))
    assert val.is_value_compatible(value.GenericValue(list, [value.UNRESOLVED_VALUE]))
    assert val.is_value_compatible(value.GenericValue(list, [TypedValue(bool)]))
    assert not val.is_value_compatible(value.GenericValue(list, [TypedValue(str)]))
    assert not val.is_value_compatible(value.GenericValue(set, [TypedValue(int)]))
    assert_eq("tuple[int, ...]", str(value.GenericValue(tuple, [TypedValue(int)])))


def test_sequence_incomplete_value():
    val = value.SequenceIncompleteValue(tuple, [TypedValue(int), TypedValue(str)])
    assert_eq("tuple[int, str]", str(val))
    assert val.is_value_compatible(TypedValue(tuple))
    assert val.is_value_compatible(
        value.GenericValue(
            tuple, [MultiValuedValue([TypedValue(int), TypedValue(str)])]
        )
    )
    assert not val.is_value_compatible(
        value.GenericValue(
            tuple, [MultiValuedValue([TypedValue(int), TypedValue(list)])]
        )
    )

    assert val.is_value_compatible(val)
    assert not val.is_value_compatible(
        value.SequenceIncompleteValue(tuple, [TypedValue(int)])
    )
    assert val.is_value_compatible(
        value.SequenceIncompleteValue(tuple, [TypedValue(bool), TypedValue(str)])
    )


def test_multi_valued_value():
    val = MultiValuedValue([TypedValue(int), KnownValue(None)])
    assert_eq("Union[int, None]", str(val))
    assert val.is_value_compatible(KnownValue(1))
    assert val.is_value_compatible(KnownValue(None))
    assert not val.is_value_compatible(KnownValue(""))
    assert not val.is_value_compatible(TypedValue(float))
    assert val.is_value_compatible(val)
    assert not val.is_value_compatible(
        MultiValuedValue([KnownValue(None), TypedValue(str)])
    )
    assert val.is_value_compatible(
        MultiValuedValue(
            [
                value.UNRESOLVED_VALUE,
                MultiValuedValue([TypedValue(int), KnownValue(None)]),
            ]
        )
    )


class ThriftEnum(object):
    X = 0
    Y = 1

    _VALUES_TO_NAMES = {0: "X", 1: "Y"}

    _NAMES_TO_VALUES = {"X": 0, "Y": 1}


def test_is_value_compatible_thrift_enum():
    val = TypedValue(ThriftEnum)
    assert val.is_value_compatible(KnownValue(0))
    assert not val.is_value_compatible(KnownValue(2))
    assert not val.is_value_compatible(KnownValue(1.0))

    assert val.is_value_compatible(TypedValue(int))
    assert val.is_value_compatible(TypedValue(ThriftEnum))
    assert not val.is_value_compatible(TypedValue(str))


def test_subclass_value():
    val = value.SubclassValue(str)
    assert_eq("Type[str]", str(val))
    assert_is(str, val.typ)
    assert val.is_type(str)
    assert not val.is_type(int)


def test_variable_name_value():
    uid_val = value.VariableNameValue(["uid", "viewer"])
    varname_map = {
        "uid": uid_val,
        "viewer": uid_val,
        "actor_id": value.VariableNameValue(["actor_id"]),
    }

    assert_is(None, value.VariableNameValue.from_varname("capybaras", varname_map))

    val = value.VariableNameValue.from_varname("uid", varname_map)
    assert_is(val, value.VariableNameValue.from_varname("viewer", varname_map))
    assert_is(val, value.VariableNameValue.from_varname("old_uid", varname_map))
    assert_is_not(val, value.VariableNameValue.from_varname("actor_id", varname_map))


def test_typeddict_value():
    val = value.TypedDictValue({"a": TypedValue(int), "b": TypedValue(str)})
    # dict iteration order in some Python versions is not deterministic
    assert_in(
        str(val), ['TypedDict({"a": int, "b": str})', 'TypedDict({"b": str, "a": int})']
    )

    assert val.is_value_compatible(value.UNRESOLVED_VALUE)
    assert val.is_value_compatible(TypedValue(dict))
    assert not val.is_value_compatible(TypedValue(str))

    # KnownValue of dict
    assert val.is_value_compatible(KnownValue({"a": 1, "b": "2"}))
    # extra keys are ok
    assert val.is_value_compatible(KnownValue({"a": 1, "b": "2", "c": 1}))
    # missing key
    assert not val.is_value_compatible(KnownValue({"a": 1}))
    # wrong type
    assert not val.is_value_compatible(KnownValue({"a": 1, "b": 2}))

    # TypedDictValue
    assert val.is_value_compatible(val)
    assert val.is_value_compatible(
        value.TypedDictValue({"a": KnownValue(1), "b": TypedValue(str)})
    )
    assert val.is_value_compatible(
        value.TypedDictValue(
            {"a": KnownValue(1), "b": TypedValue(str), "c": TypedValue(float)}
        )
    )
    assert not val.is_value_compatible(
        value.TypedDictValue({"a": KnownValue(1), "b": TypedValue(int)})
    )
    assert not val.is_value_compatible(value.TypedDictValue({"b": TypedValue(str)}))

    # DictIncompleteValue
    assert val.is_value_compatible(
        value.DictIncompleteValue(
            [(KnownValue("a"), TypedValue(int)), (KnownValue("b"), TypedValue(str))]
        )
    )
    assert val.is_value_compatible(
        value.DictIncompleteValue(
            [
                (KnownValue("a"), TypedValue(int)),
                (KnownValue("b"), TypedValue(str)),
                (KnownValue("c"), value.UNRESOLVED_VALUE),
            ]
        )
    )
    assert val.is_value_compatible(
        value.DictIncompleteValue(
            [
                (KnownValue("a"), TypedValue(int)),
                (value.UNRESOLVED_VALUE, TypedValue(str)),
            ]
        )
    )
    assert not val.is_value_compatible(
        value.DictIncompleteValue([(value.UNRESOLVED_VALUE, TypedValue(str))])
    )
    assert not val.is_value_compatible(
        value.DictIncompleteValue(
            [(KnownValue("a"), TypedValue(int)), (KnownValue("b"), TypedValue(float))]
        )
    )


@skip_before((3, 5))
def test_new_type_value():
    from typing import NewType

    nt1 = NewType("nt1", int)
    nt1_val = value.NewTypeValue(nt1)
    nt2 = NewType("nt2", int)
    nt2_val = value.NewTypeValue(nt2)
    assert not nt1_val.is_value_compatible(nt2_val)
    # This should eventually return False
    assert nt1_val.is_value_compatible(TypedValue(int))
    assert TypedValue(int).is_value_compatible(nt1_val)
