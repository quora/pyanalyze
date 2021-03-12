from .annotations import type_from_value
from .error_code import ErrorCode
from .find_unused import used
from .format_strings import parse_format_string
from .safe import safe_hasattr
from .stacked_scopes import (
    NULL_CONSTRAINT,
    AbstractConstraint,
    Constraint,
    ConstraintType,
    OrConstraint,
)
from .signature import (
    ExtendedArgSpec,
    Parameter,
    ImplementationFnReturn,
    VarsDict,
    clean_up_implementation_fn_return,
)
from .value import (
    TypedValue,
    SubclassValue,
    GenericValue,
    DictIncompleteValue,
    SequenceIncompleteValue,
    TypedDictValue,
    KnownValue,
    MultiValuedValue,
    TypeVarValue,
    UNRESOLVED_VALUE,
    Value,
    unite_values,
    flatten_values,
)

import ast
from functools import reduce
import collections.abc
from itertools import product
import qcore
import inspect
import warnings
from typing import cast, NewType, TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from .name_check_visitor import NameCheckVisitor

_NO_ARG_SENTINEL = qcore.MarkerObject("no argument given")

T = TypeVar("T")
IterableValue = GenericValue(collections.abc.Iterable, [TypeVarValue(T)])


@used  # exposed as an API
def assert_is_value(obj: object, value: Value) -> None:
    """Used to test test_scope's value inference.

    Takes two arguments: a Python object and a Value object. This function does nothing at runtime,
    but test_scope checks that when it encounters a call to assert_is_value, the inferred value of
    the object matches that in the call.

    """
    pass


@used  # exposed as an API
def dump_value(value: object) -> None:
    """Used for debugging test_scope.

    Calling it will make test_scope print out the argument's inferred value. Does nothing at
    runtime.

    """
    pass


def _maybe_or_constraint(
    left: AbstractConstraint, right: AbstractConstraint
) -> AbstractConstraint:
    if left is NULL_CONSTRAINT or right is NULL_CONSTRAINT:
        return NULL_CONSTRAINT
    return OrConstraint(left, right)


def flatten_unions(
    callable: Callable[..., ImplementationFnReturn], *values: Value
) -> ImplementationFnReturn:
    value_lists = [flatten_values(val) for val in values]
    return_values, constraints, no_return_unless = zip(
        *[
            clean_up_implementation_fn_return(callable(*vals))
            for vals in product(*value_lists)
        ]
    )
    return (
        unite_values(*return_values),
        reduce(_maybe_or_constraint, constraints),
        reduce(_maybe_or_constraint, no_return_unless),
    )


def replace_known_sequence_value(value: Value) -> Value:
    if isinstance(value, KnownValue):
        if isinstance(value.val, (list, tuple, set)):
            return SequenceIncompleteValue(
                type(value.val), [KnownValue(elt) for elt in value.val]
            )
        elif isinstance(value.val, dict):
            return DictIncompleteValue(
                [(KnownValue(k), KnownValue(v)) for k, v in value.val.items()]
            )
    return value


# Implementations of some important functions for use in their ExtendedArgSpecs (see above). These
# are called when the test_scope checker encounters call to these functions.
def _isinstance_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    class_or_tuple = variables["class_or_tuple"]
    if not isinstance(class_or_tuple, KnownValue):
        return TypedValue(bool), NULL_CONSTRAINT
    if len(node.args) < 1:
        return TypedValue(bool), NULL_CONSTRAINT
    varname = visitor.varname_for_constraint(node.args[0])
    if varname is None:
        return TypedValue(bool), NULL_CONSTRAINT
    if isinstance(class_or_tuple.val, type):
        return (
            TypedValue(bool),
            Constraint(varname, ConstraintType.is_instance, True, class_or_tuple.val),
        )
    elif isinstance(class_or_tuple.val, tuple) and all(
        isinstance(elt, type) for elt in class_or_tuple.val
    ):
        constraints = [
            Constraint(varname, ConstraintType.is_instance, True, elt)
            for elt in class_or_tuple.val
        ]
        return TypedValue(bool), reduce(OrConstraint, constraints)
    else:
        return TypedValue(bool), NULL_CONSTRAINT


def _hasattr_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    obj = variables["object"]
    name = variables["name"]
    if not isinstance(name, KnownValue):
        return TypedValue(bool)
    if not isinstance(obj, (TypedValue, KnownValue)):
        return TypedValue(bool)

    typ = obj.typ if isinstance(obj, TypedValue) else type(obj.val)
    # interpret a hasattr check as a sign that the object (somehow) has the attribute
    visitor._record_type_attr_set(typ, name.val, node, UNRESOLVED_VALUE)

    # if the value exists on the type or instance, hasattr should return True
    # don't interpret the opposite to mean it should return False, as the attribute may
    # exist on a child class or get assigned at runtime
    if isinstance(obj, TypedValue) and safe_hasattr(obj.typ, name.val):
        return KnownValue(True)
    elif isinstance(obj, KnownValue) and safe_hasattr(obj.val, name.val):
        return KnownValue(True)
    else:
        return TypedValue(bool)


def _setattr_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    # if we set an attribute on a value of known type, record it to the attribute checker so we
    # don't say the attribute is undefined
    obj = variables["object"]
    name = variables["name"]
    if isinstance(obj, TypedValue):
        typ = obj.typ
        if isinstance(name, KnownValue):
            visitor._record_type_attr_set(typ, name.val, node, variables["value"])
        else:
            visitor._record_type_has_dynamic_attrs(typ)
    return KnownValue(None)


def _super_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    typ = variables["type"]
    obj = variables["obj"]
    if typ == KnownValue(None):
        # Zero-argument super()
        if visitor.in_comprehension_body:
            visitor.show_error(
                node,
                "Zero-argument super() does not work inside a comprehension",
                ErrorCode.bad_super_call,
            )
        elif visitor.scopes.is_nested_function():
            visitor.show_error(
                node,
                "Zero-argument super() does not work inside a nested function",
                ErrorCode.bad_super_call,
            )
        current_class = visitor.asynq_checker.current_class
        if current_class is not None:
            try:
                first_arg = visitor.scopes.get("%first_arg", None, visitor.state)
            except KeyError:
                # something weird with this function; give up
                visitor.show_error(
                    node, "failed to find %first_arg", ErrorCode.bad_super_call
                )
                return UNRESOLVED_VALUE
            else:
                if isinstance(first_arg, SubclassValue):
                    return KnownValue(super(current_class, first_arg.typ))
                elif isinstance(first_arg, KnownValue):
                    return KnownValue(super(current_class, first_arg.val))
                elif isinstance(first_arg, TypedValue):
                    return TypedValue(super(current_class, first_arg.typ))
                else:
                    return UNRESOLVED_VALUE
        return UNRESOLVED_VALUE

    if isinstance(typ, KnownValue):
        if inspect.isclass(typ.val):
            cls = typ.val
        else:
            visitor.show_error(
                node,
                "First argument to super must be a class",
                ErrorCode.bad_super_call,
            )
            return UNRESOLVED_VALUE
    else:
        return UNRESOLVED_VALUE  # probably a dynamically created class

    if isinstance(obj, TypedValue) and obj.typ is not type:
        instance_type = obj.typ
        is_value = True
    elif isinstance(obj, SubclassValue):
        instance_type = obj.typ
        is_value = False
    else:
        return UNRESOLVED_VALUE

    if not issubclass(instance_type, cls):
        visitor.show_error(
            node, "Incompatible arguments to super", ErrorCode.bad_super_call
        )

    current_class = visitor.asynq_checker.current_class
    if current_class is not None and cls is not current_class:
        visitor.show_error(
            node,
            "First argument to super() is not the current class",
            ErrorCode.bad_super_call,
        )

    try:
        super_val = super(cls, instance_type)
    except Exception:
        visitor.show_error(node, "Bad arguments to super", ErrorCode.bad_super_call)
        return UNRESOLVED_VALUE

    if is_value:
        return TypedValue(super_val)
    else:
        return KnownValue(super_val)


def _tuple_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    return _sequence_impl(tuple, variables, visitor, node)


def _list_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    return _sequence_impl(list, variables, visitor, node)


def _set_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    return _sequence_impl(set, variables, visitor, node)


def _sequence_impl(
    typ: type, variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    iterable = variables["iterable"]
    if iterable == KnownValue(_NO_ARG_SENTINEL):
        return KnownValue(typ())
    elif isinstance(iterable, KnownValue):
        try:
            return KnownValue(typ(iterable.val))
        except TypeError:
            if iterable.val is not None:
                visitor.show_error(
                    node,
                    f"Object {iterable.val!r} is not iterable",
                    ErrorCode.unsupported_operation,
                )
            return TypedValue(typ)
    elif isinstance(iterable, SequenceIncompleteValue):
        return SequenceIncompleteValue(typ, iterable.members)
    elif isinstance(iterable, DictIncompleteValue):
        return SequenceIncompleteValue(typ, [key for key, _ in iterable.items])
    else:
        tv_map = IterableValue.can_assign(iterable, visitor)
        if tv_map is None:
            visitor.show_error(
                node,
                f"Object of type {iterable.typ} is not iterable",
                ErrorCode.unsupported_operation,
            )
        elif T in tv_map:
            return GenericValue(typ, [tv_map[T]])
        return TypedValue(typ)


def _list_append_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.Call
) -> ImplementationFnReturn:
    lst = replace_known_sequence_value(variables["self"])
    element = variables["object"]
    varname = visitor.varname_for_self_constraint(node)
    if isinstance(lst, SequenceIncompleteValue):
        no_return_unless = Constraint(
            varname,
            ConstraintType.is_value_object,
            True,
            SequenceIncompleteValue(list, (*lst.members, element)),
        )
        return KnownValue(None), NULL_CONSTRAINT, no_return_unless
    elif isinstance(lst, GenericValue):
        expected_type = lst.get_generic_arg_for_type(list, visitor, 0)
        if not expected_type.is_assignable(element, visitor):
            visitor.show_error(
                node,
                f"Cannot append value of type {element} to list of {expected_type}",
                ErrorCode.incompatible_argument,
            )
    return KnownValue(None)


def _dict_setitem_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    self_value = replace_known_sequence_value(variables["self"])
    key = variables["k"]
    value = variables["v"]
    # apparently for a[b] = c we get passed the AST node for a
    varname = visitor.varname_for_constraint(node)
    if isinstance(self_value, TypedDictValue):
        if not isinstance(key, KnownValue) or not isinstance(key.val, str):
            visitor.show_error(
                node,
                f"TypedDict key must be a string literal (got {key})",
                ErrorCode.invalid_typeddict_key,
            )
        elif key.val not in self_value.items:
            visitor.show_error(
                node,
                f"Key {key.val!r} does not exist in {self_value}",
                ErrorCode.invalid_typeddict_key,
            )
        else:
            expected_type = self_value.items[key.val]
            if not expected_type.is_assignable(value, visitor):
                visitor.show_error(
                    node,
                    f"Value for key {key.val!r} must be {expected_type}, not {value}",
                    ErrorCode.incompatible_argument,
                )
        return KnownValue(None)
    elif isinstance(self_value, DictIncompleteValue):
        no_return_unless = Constraint(
            varname,
            ConstraintType.is_value_object,
            True,
            # This might create a duplicate but searching for that would
            # be O(n^2) and doesn't seem too useful.
            DictIncompleteValue([*self_value.items, (key, value)]),
        )
        return KnownValue(None), NULL_CONSTRAINT, no_return_unless
    elif isinstance(self_value, TypedValue):
        key_type = self_value.get_generic_arg_for_type(dict, visitor, 0)
        if not key_type.is_assignable(key, visitor):
            visitor.show_error(
                node,
                f"Cannot set key of type {key} (expecting {key_type})",
                ErrorCode.incompatible_argument,
            )
        value_type = self_value.get_generic_arg_for_type(dict, visitor, 1)
        if not value_type.is_assignable(value, visitor):
            visitor.show_error(
                node,
                f"Cannot set value of type {value} (expecting {value_type})",
                ErrorCode.incompatible_argument,
            )
    return KnownValue(None)


def _dict_getitem_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    def inner(key: Value) -> Value:
        self_value = variables["self"]
        if isinstance(key, KnownValue):
            try:
                hash(key.val)
            except Exception:
                visitor.show_error(
                    node,
                    f"Dictionary key {key} is not hashable",
                    ErrorCode.unhashable_key,
                )
                return UNRESOLVED_VALUE
        if isinstance(self_value, KnownValue):
            if isinstance(key, KnownValue):
                try:
                    return_value = self_value.val[key.val]
                except Exception:
                    # No error here, the key may have been added where we couldn't see it.
                    return UNRESOLVED_VALUE
                else:
                    return KnownValue(return_value)
            # else just treat it together with DictIncompleteValue
            self_value = replace_known_sequence_value(self_value)
        if isinstance(self_value, TypedDictValue):
            if not TypedValue(str).is_assignable(key, visitor):
                visitor.show_error(
                    node,
                    f"TypedDict key must be str, not {key}",
                    ErrorCode.invalid_typeddict_key,
                )
                return UNRESOLVED_VALUE
            elif isinstance(key, KnownValue):
                try:
                    return self_value.items[key.val]
                # probably KeyError, but catch anything in case it's an
                # unhashable str subclass or something
                except Exception:
                    # No error here; TypedDicts may have additional keys at runtime.
                    pass
            # TODO strictly we should throw an error for any non-Literal or unknown key:
            # https://www.python.org/dev/peps/pep-0589/#supported-and-unsupported-operations
            # Don't do that yet because it may cause too much disruption.
            return UNRESOLVED_VALUE
        elif isinstance(self_value, DictIncompleteValue):
            possible_values = [
                dict_value
                for dict_key, dict_value in self_value.items
                if dict_key.is_assignable(key, visitor)
            ]
            if not possible_values:
                # No error here, the key may have been added where we couldn't see it.
                return UNRESOLVED_VALUE
            return unite_values(*possible_values)
        elif isinstance(self_value, TypedValue):
            return self_value.get_generic_arg_for_type(dict, visitor, 1)
        else:
            return UNRESOLVED_VALUE

    return flatten_unions(inner, variables["k"])


def _list_add_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.Call
) -> ImplementationFnReturn:
    def inner(left: Value, right: Value) -> ImplementationFnReturn:
        left = replace_known_sequence_value(left)
        right = replace_known_sequence_value(right)
        if isinstance(left, SequenceIncompleteValue) and isinstance(
            right, SequenceIncompleteValue
        ):
            return SequenceIncompleteValue(list, [*left.members, *right.members])
        elif isinstance(left, TypedValue) and isinstance(right, TypedValue):
            left_args = left.get_generic_args_for_type(list, visitor)
            left_arg = left_args[0] if left_args else UNRESOLVED_VALUE
            right_args = right.get_generic_args_for_type(list, visitor)
            right_arg = right_args[0] if right_args else UNRESOLVED_VALUE
            return GenericValue(list, [unite_values(left_arg, right_arg)])
        else:
            return TypedValue(list)

    return flatten_unions(inner, variables["self"], variables["x"])


def _list_extend_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.Call
) -> ImplementationFnReturn:
    varname = visitor.varname_for_self_constraint(node)

    def inner(lst: Value, iterable: Value) -> ImplementationFnReturn:
        lst = replace_known_sequence_value(lst)
        iterable = replace_known_sequence_value(iterable)
        if isinstance(lst, SequenceIncompleteValue):
            if isinstance(iterable, SequenceIncompleteValue) and issubclass(
                iterable.typ, (list, tuple)
            ):
                constrained_value = SequenceIncompleteValue(
                    list, (*lst.members, *iterable.members)
                )
            else:
                if isinstance(iterable, TypedValue):
                    arg_type = iterable.get_generic_arg_for_type(
                        collections.abc.Iterable, visitor, 0
                    )
                else:
                    arg_type = UNRESOLVED_VALUE
                constrained_value = GenericValue(
                    list, [unite_values(*lst.members, arg_type)]
                )
            no_return_unless = Constraint(
                varname, ConstraintType.is_value_object, True, constrained_value
            )
            return KnownValue(None), NULL_CONSTRAINT, no_return_unless
        elif isinstance(lst, GenericValue):
            expected_type = lst.get_generic_arg_for_type(list, visitor, 0)
            if isinstance(iterable, TypedValue):
                actual_type = iterable.get_generic_arg_for_type(
                    collections.abc.Iterable, visitor, 0
                )
                if not expected_type.is_assignable(actual_type, visitor):
                    visitor.show_error(
                        node,
                        f"Cannot extend list of {expected_type} with values of type {actual_type}",
                        ErrorCode.incompatible_argument,
                    )
        return KnownValue(None)

    return flatten_unions(inner, variables["self"], variables["iterable"])


def _set_add_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.Call
) -> ImplementationFnReturn:
    set_value = replace_known_sequence_value(variables["self"])
    element = variables["object"]
    varname = visitor.varname_for_self_constraint(node)
    if isinstance(set_value, SequenceIncompleteValue):
        no_return_unless = Constraint(
            varname,
            ConstraintType.is_value_object,
            True,
            SequenceIncompleteValue(set, (*set_value.members, element)),
        )
        return KnownValue(None), NULL_CONSTRAINT, no_return_unless
    elif isinstance(set_value, GenericValue):
        set_args = set_value.get_generic_args_for_type(set, visitor)
        if set_args:
            expected_type = set_args[0]
            if not expected_type.is_assignable(element, visitor):
                visitor.show_error(
                    node,
                    f"Cannot add value of type {element} to set of {expected_type}",
                    ErrorCode.incompatible_argument,
                )
    return KnownValue(None)


def _assert_is_value_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    if not visitor._is_checking():
        return KnownValue(None)
    obj = variables["obj"]
    expected_value = variables["value"]
    if not isinstance(expected_value, KnownValue):
        visitor.show_error(
            node,
            f"Value argument to assert_is_value must be a KnownValue (got {expected_value}) {obj}",
            ErrorCode.inference_failure,
        )
    else:
        if obj != expected_value.val:
            visitor.show_error(
                node,
                f"Bad value inference: expected {expected_value.val}, got {obj}",
                ErrorCode.inference_failure,
            )
    return KnownValue(None)


def _dump_value_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    if visitor._is_checking():
        visitor.show_error(
            node, f"Value: {variables['value']}", ErrorCode.inference_failure
        )
    return KnownValue(None)


def _str_format_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    self = variables["self"]
    if not isinstance(self, KnownValue):
        return TypedValue(str)
    args = variables["args"]
    kwargs = variables["kwargs"]
    template = self.val
    used_indices = set()
    used_kwargs = set()
    current_index = 0
    parsed, errors = parse_format_string(template)
    if errors:
        _, message = errors[0]
        visitor.show_error(node, message, error_code=ErrorCode.incompatible_call)
        return TypedValue(str)
    for field in parsed.iter_replacement_fields():
        # TODO validate conversion specifiers, attributes, etc.
        if field.arg_name is None:
            if current_index >= len(args):
                visitor.show_error(
                    node,
                    "Too few arguments to format string (expected at least %s)"
                    % (current_index,),
                    error_code=ErrorCode.incompatible_call,
                )
            used_indices.add(current_index)
            current_index += 1
        elif isinstance(field.arg_name, int):
            index = field.arg_name
            if index >= len(args):
                visitor.show_error(
                    node,
                    "Numbered argument %s to format string is out of range" % (index,),
                    error_code=ErrorCode.incompatible_call,
                )
            used_indices.add(index)
        else:
            if field.arg_name not in kwargs:
                visitor.show_error(
                    node,
                    "Named argument %s to format string was not given"
                    % (field.arg_name,),
                    error_code=ErrorCode.incompatible_call,
                )
            used_kwargs.add(field.arg_name)
    unused_indices = set(range(len(args))) - used_indices
    if unused_indices:
        visitor.show_error(
            node,
            "Numbered argument(s) %s were not used"
            % (", ".join(map(str, sorted(unused_indices)))),
            error_code=ErrorCode.incompatible_call,
        )
    unused_kwargs = set(kwargs) - used_kwargs
    if unused_kwargs:
        visitor.show_error(
            node,
            "Named argument(s) %s were not used" % (", ".join(sorted(unused_kwargs))),
            error_code=ErrorCode.incompatible_call,
        )
    return TypedValue(str)


def _cast_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> Value:
    typ = variables["typ"]
    return type_from_value(typ, visitor=visitor, node=node)


def _subclasses_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> Value:
    """Overridden because typeshed types make it (T) => List[T] instead."""
    self_obj = variables["self"]
    if isinstance(self_obj, KnownValue) and isinstance(self_obj.val, type):
        return KnownValue(self_obj.val.__subclasses__())
    return GenericValue(list, [TypedValue(type)])


def _assert_is_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    return _qcore_assert_impl(variables, visitor, node, ConstraintType.is_value, True)


def _assert_is_not_impl(
    variables: VarsDict, visitor: "NameCheckVisitor", node: ast.AST
) -> ImplementationFnReturn:
    return _qcore_assert_impl(variables, visitor, node, ConstraintType.is_value, False)


def _qcore_assert_impl(
    variables: VarsDict,
    visitor: "NameCheckVisitor",
    node: ast.AST,
    constraint_type: ConstraintType,
    positive: bool,
) -> ImplementationFnReturn:
    if len(node.args) < 2:
        # arguments were passed as kwargs
        return KnownValue(None), NULL_CONSTRAINT, NULL_CONSTRAINT
    left_varname = visitor.varname_for_constraint(node.args[0])
    right_varname = visitor.varname_for_constraint(node.args[1])
    if left_varname is not None and isinstance(variables["actual"], KnownValue):
        varname = left_varname
        constrained_to = variables["actual"].val
    elif right_varname is not None and isinstance(variables["expected"], KnownValue):
        varname = right_varname
        constrained_to = variables["expected"].val
    else:
        return KnownValue(None), NULL_CONSTRAINT, NULL_CONSTRAINT

    no_return_unless = Constraint(varname, constraint_type, positive, constrained_to)
    return KnownValue(None), NULL_CONSTRAINT, no_return_unless


_ENCODING_PARAMETER = Parameter("encoding", typ=TypedValue(str), default_value="")


def get_default_argspecs():
    return {
        assert_is_value: ExtendedArgSpec(
            [Parameter("obj"), Parameter("value", typ=TypedValue(Value))],
            implementation=_assert_is_value_impl,
            name="assert_is_value",
        ),
        dump_value: ExtendedArgSpec(
            [Parameter("value")], implementation=_dump_value_impl, name="dump_value"
        ),
        # builtins
        isinstance: ExtendedArgSpec(
            [Parameter("obj"), Parameter("class_or_tuple")],
            name="isinstance",
            implementation=_isinstance_impl,
        ),
        getattr: ExtendedArgSpec(
            [
                Parameter("object"),
                Parameter("name", typ=TypedValue(str)),
                Parameter("default", default_value=None),
            ],
            name="getattr",
        ),
        hasattr: ExtendedArgSpec(
            [Parameter("object"), Parameter("name", typ=TypedValue(str))],
            return_value=TypedValue(bool),
            name="hasattr",
            implementation=_hasattr_impl,
        ),
        setattr: ExtendedArgSpec(
            [
                Parameter("object"),
                Parameter("name", typ=TypedValue(str)),
                Parameter("value"),
            ],
            return_value=KnownValue(None),
            name="setattr",
            implementation=_setattr_impl,
        ),
        super: ExtendedArgSpec(
            [
                Parameter(
                    "type",
                    default_value=None,
                ),
                Parameter("obj", default_value=None),
            ],
            name="super",
            implementation=_super_impl,
        ),
        tuple: ExtendedArgSpec(
            [Parameter("iterable", default_value=_NO_ARG_SENTINEL)],
            name="tuple",
            implementation=_tuple_impl,
        ),
        list: ExtendedArgSpec(
            [Parameter("iterable", default_value=_NO_ARG_SENTINEL)],
            name="list",
            implementation=_list_impl,
        ),
        list.append: ExtendedArgSpec(
            [Parameter("self", typ=TypedValue(list)), Parameter("object")],
            name="list.append",
            implementation=_list_append_impl,
        ),
        list.__add__: ExtendedArgSpec(
            [
                Parameter("self", typ=TypedValue(list)),
                Parameter("x", typ=TypedValue(list)),
            ],
            name="list.__add__",
            implementation=_list_add_impl,
        ),
        list.extend: ExtendedArgSpec(
            [
                Parameter("self", typ=TypedValue(list)),
                Parameter("iterable", typ=TypedValue(collections.abc.Iterable)),
            ],
            name="list.extend",
            implementation=_list_extend_impl,
        ),
        set.add: ExtendedArgSpec(
            [Parameter("self", typ=TypedValue(set)), Parameter("object")],
            name="set.add",
            implementation=_set_add_impl,
        ),
        dict.__setitem__: ExtendedArgSpec(
            [Parameter("self", typ=TypedValue(dict)), Parameter("k"), Parameter("v")],
            name="dict.__setitem__",
            implementation=_dict_setitem_impl,
        ),
        dict.__getitem__: ExtendedArgSpec(
            [Parameter("self", typ=TypedValue(dict)), Parameter("k")],
            name="dict.__getitem__",
            implementation=_dict_getitem_impl,
        ),
        set: ExtendedArgSpec(
            [Parameter("iterable", default_value=_NO_ARG_SENTINEL)],
            name="set",
            implementation=_set_impl,
        ),
        bytes.decode: ExtendedArgSpec(
            [
                Parameter("self", typ=TypedValue(bytes)),
                _ENCODING_PARAMETER,
                Parameter("errors", typ=TypedValue(str), default_value=""),
            ],
            name="bytes.decode",
            return_value=TypedValue(str),
        ),
        str.encode: ExtendedArgSpec(
            [
                Parameter("self", typ=TypedValue(str)),
                _ENCODING_PARAMETER,
                Parameter("errors", typ=TypedValue(str), default_value=""),
            ],
            name="str.encode",
            return_value=TypedValue(bytes),
        ),
        str.format: ExtendedArgSpec(
            [Parameter("self", typ=TypedValue(str))],
            starargs="args",
            kwargs="kwargs",
            name="str.format",
            implementation=_str_format_impl,
        ),
        cast: ExtendedArgSpec(
            [Parameter("typ"), Parameter("val")],
            name="typing.cast",
            implementation=_cast_impl,
        ),
        # workaround for https://github.com/python/typeshed/pull/3501
        warnings.warn: ExtendedArgSpec(
            [
                Parameter(
                    "message",
                    typ=MultiValuedValue([TypedValue(str), TypedValue(Warning)]),
                ),
                Parameter("category", typ=UNRESOLVED_VALUE, default_value=None),
                Parameter("stacklevel", typ=TypedValue(int), default_value=1),
            ],
            name="warnings.warn",
            return_value=KnownValue(None),
        ),
        # qcore/asynq
        # just so we can infer the return value
        qcore.utime: ExtendedArgSpec([], name="utime", return_value=TypedValue(int)),
        qcore.asserts.assert_is: ExtendedArgSpec(
            [
                Parameter("expected"),
                Parameter("actual"),
                Parameter("message", default_value=None),
                Parameter("extra", default_value=None),
            ],
            name="assert_is",
            implementation=_assert_is_impl,
        ),
        qcore.asserts.assert_is_not: ExtendedArgSpec(
            [
                Parameter("expected"),
                Parameter("actual"),
                Parameter("message", default_value=None),
                Parameter("extra", default_value=None),
            ],
            name="assert_is_not",
            implementation=_assert_is_not_impl,
        ),
        # Need to override this because the type for the tp parameter in typeshed is too strict
        NewType: ExtendedArgSpec(
            [Parameter("name", typ=TypedValue(str)), Parameter(name="tp")],
            name="NewType",
        ),
        type.__subclasses__: ExtendedArgSpec(
            [Parameter("self")],
            name="type.__subclasses__",
            implementation=_subclasses_impl,
        ),
    }
