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
    PredicateProvider,
    OrConstraint,
    Varname,
)
from .signature import SigParameter, Signature, ImplReturn, CallContext
from .value import (
    AnnotatedValue,
    CanAssignError,
    HasAttrGuardExtension,
    ParameterTypeGuardExtension,
    TypedValue,
    SubclassValue,
    GenericValue,
    NewTypeValue,
    DictIncompleteValue,
    SequenceIncompleteValue,
    TypedDictValue,
    KnownValue,
    MultiValuedValue,
    TypeVarValue,
    UNRESOLVED_VALUE,
    NO_RETURN_VALUE,
    KNOWN_MUTABLE_TYPES,
    Value,
    unite_values,
    flatten_values,
    replace_known_sequence_value,
)

from functools import reduce
import collections.abc
from itertools import product
import qcore
import inspect
import warnings
from typing import cast, Dict, NewType, Callable, TypeVar, Optional, Union

_NO_ARG_SENTINEL = KnownValue(qcore.MarkerObject("no argument given"))

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


def clean_up_implementation_fn_return(
    return_value: Union[Value, ImplReturn]
) -> ImplReturn:
    if isinstance(return_value, Value):
        return ImplReturn(return_value)
    return return_value


def flatten_unions(
    callable: Callable[..., Union[ImplReturn, Value]], *values: Value
) -> ImplReturn:
    value_lists = [flatten_values(val) for val in values]
    results = [
        clean_up_implementation_fn_return(callable(*vals))
        for vals in product(*value_lists)
    ]
    if not results:
        return ImplReturn(NO_RETURN_VALUE)
    return_values, constraints, no_return_unless = zip(*results)
    return ImplReturn(
        unite_values(*return_values),
        reduce(_maybe_or_constraint, constraints),
        reduce(_maybe_or_constraint, no_return_unless),
    )


# Implementations of some important functions for use in their ExtendedArgSpecs (see above). These
# are called when the test_scope checker encounters call to these functions.


def _issubclass_impl(ctx: CallContext) -> Value:
    class_or_tuple = ctx.vars["class_or_tuple"]
    extension = None
    if isinstance(class_or_tuple, KnownValue):
        if isinstance(class_or_tuple.val, type):
            extension = ParameterTypeGuardExtension(
                "cls", SubclassValue(TypedValue(class_or_tuple.val))
            )
        elif isinstance(class_or_tuple.val, tuple) and all(
            isinstance(elt, type) for elt in class_or_tuple.val
        ):
            vals = [SubclassValue(TypedValue(elt)) for elt in class_or_tuple.val]
            extension = ParameterTypeGuardExtension("cls", MultiValuedValue(vals))
    if extension is not None:
        return AnnotatedValue(TypedValue(bool), [extension])
    return TypedValue(bool)


def _isinstance_impl(ctx: CallContext) -> ImplReturn:
    class_or_tuple = ctx.vars["class_or_tuple"]
    varname = ctx.varname_for_arg("obj")
    return ImplReturn(
        TypedValue(bool), _constraint_from_isinstance(varname, class_or_tuple)
    )


def _constraint_from_isinstance(
    varname: Optional[Varname], class_or_tuple: Value
) -> AbstractConstraint:
    if varname is None:
        return NULL_CONSTRAINT
    if not isinstance(class_or_tuple, KnownValue):
        return NULL_CONSTRAINT

    if isinstance(class_or_tuple.val, type):
        return Constraint(varname, ConstraintType.is_instance, True, class_or_tuple.val)
    elif isinstance(class_or_tuple.val, tuple) and all(
        isinstance(elt, type) for elt in class_or_tuple.val
    ):
        constraints = [
            Constraint(varname, ConstraintType.is_instance, True, elt)
            for elt in class_or_tuple.val
        ]
        return reduce(OrConstraint, constraints)
    else:
        return NULL_CONSTRAINT


def _assert_is_instance_impl(ctx: CallContext) -> ImplReturn:
    class_or_tuple = ctx.vars["types"]
    varname = ctx.varname_for_arg("value")
    return ImplReturn(
        KnownValue(None),
        NULL_CONSTRAINT,
        _constraint_from_isinstance(varname, class_or_tuple),
    )


def _hasattr_impl(ctx: CallContext) -> Value:
    obj = ctx.vars["object"]
    name = ctx.vars["name"]
    if not isinstance(name, KnownValue) or not isinstance(name.val, str):
        return TypedValue(bool)
    if not isinstance(obj, (TypedValue, KnownValue)):
        return_value = TypedValue(bool)
    else:
        typ = obj.typ if isinstance(obj, TypedValue) else type(obj.val)
        # interpret a hasattr check as a sign that the object (somehow) has the attribute
        ctx.visitor._record_type_attr_set(typ, name.val, ctx.node, UNRESOLVED_VALUE)

        # if the value exists on the type or instance, hasattr should return True
        # don't interpret the opposite to mean it should return False, as the attribute may
        # exist on a child class or get assigned at runtime
        if isinstance(obj, TypedValue) and safe_hasattr(obj.typ, name.val):
            return_value = KnownValue(True)
        elif isinstance(obj, KnownValue) and safe_hasattr(obj.val, name.val):
            return_value = KnownValue(True)
        else:
            return_value = TypedValue(bool)
    metadata = [HasAttrGuardExtension("object", name, UNRESOLVED_VALUE)]
    return AnnotatedValue(return_value, metadata)


def _setattr_impl(ctx: CallContext) -> Value:
    # if we set an attribute on a value of known type, record it to the attribute checker so we
    # don't say the attribute is undefined
    obj = ctx.vars["object"]
    name = ctx.vars["name"]
    if isinstance(obj, TypedValue):
        typ = obj.typ
        if isinstance(name, KnownValue):
            ctx.visitor._record_type_attr_set(
                typ, name.val, ctx.node, ctx.vars["value"]
            )
        else:
            ctx.visitor._record_type_has_dynamic_attrs(typ)
    return KnownValue(None)


def _super_impl(ctx: CallContext) -> Value:
    typ = ctx.vars["type"]
    obj = ctx.vars["obj"]
    if typ is _NO_ARG_SENTINEL:
        # Zero-argument super()
        if ctx.visitor.in_comprehension_body:
            ctx.show_error(
                "Zero-argument super() does not work inside a comprehension",
                ErrorCode.bad_super_call,
            )
        elif ctx.visitor.scopes.is_nested_function():
            ctx.show_error(
                "Zero-argument super() does not work inside a nested function",
                ErrorCode.bad_super_call,
            )
        current_class = ctx.visitor.asynq_checker.current_class
        if current_class is not None:
            try:
                first_arg = ctx.visitor.scopes.get(
                    "%first_arg", None, ctx.visitor.state
                )
            except KeyError:
                # something weird with this function; give up
                ctx.show_error("failed to find %first_arg", ErrorCode.bad_super_call)
                return UNRESOLVED_VALUE
            else:
                if isinstance(first_arg, SubclassValue) and isinstance(
                    first_arg.typ, TypedValue
                ):
                    return KnownValue(super(current_class, first_arg.typ.typ))
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
            ctx.show_error(
                "First argument to super must be a class", ErrorCode.bad_super_call
            )
            return UNRESOLVED_VALUE
    else:
        return UNRESOLVED_VALUE  # probably a dynamically created class

    if isinstance(obj, TypedValue) and obj.typ is not type:
        instance_type = obj.typ
        is_value = True
    elif isinstance(obj, SubclassValue) and isinstance(obj.typ, TypedValue):
        instance_type = obj.typ.typ
        is_value = False
    else:
        return UNRESOLVED_VALUE

    if not issubclass(instance_type, cls):
        ctx.show_error("Incompatible arguments to super", ErrorCode.bad_super_call)

    current_class = ctx.visitor.asynq_checker.current_class
    if current_class is not None and cls is not current_class:
        ctx.show_error(
            "First argument to super() is not the current class",
            ErrorCode.bad_super_call,
        )

    try:
        super_val = super(cls, instance_type)
    except Exception:
        ctx.show_error("Bad arguments to super", ErrorCode.bad_super_call)
        return UNRESOLVED_VALUE

    if is_value:
        return TypedValue(super_val)
    else:
        return KnownValue(super_val)


def _tuple_impl(ctx: CallContext) -> Value:
    return _sequence_impl(tuple, ctx)


def _list_impl(ctx: CallContext) -> Value:
    return _sequence_impl(list, ctx)


def _set_impl(ctx: CallContext) -> Value:
    return _sequence_impl(set, ctx)


def _sequence_impl(typ: type, ctx: CallContext) -> Value:
    iterable = ctx.vars["iterable"]
    if iterable is _NO_ARG_SENTINEL:
        return KnownValue(typ())
    elif isinstance(iterable, KnownValue):
        try:
            return KnownValue(typ(iterable.val))
        except TypeError:
            if iterable.val is not None:
                ctx.show_error(
                    f"Object {iterable.val!r} is not iterable",
                    ErrorCode.unsupported_operation,
                    arg="iterable",
                )
            return TypedValue(typ)
    elif isinstance(iterable, SequenceIncompleteValue):
        return SequenceIncompleteValue(typ, iterable.members)
    elif isinstance(iterable, DictIncompleteValue):
        return SequenceIncompleteValue(typ, [key for key, _ in iterable.items])
    else:
        tv_map = IterableValue.can_assign(iterable, ctx.visitor)
        if isinstance(tv_map, CanAssignError):
            ctx.show_error(
                f"{iterable} is not iterable",
                ErrorCode.unsupported_operation,
                arg="iterable",
                detail=str(tv_map),
            )
        elif T in tv_map:
            return GenericValue(typ, [tv_map[T]])
        return TypedValue(typ)


def _list_append_impl(ctx: CallContext) -> ImplReturn:
    lst = replace_known_sequence_value(ctx.vars["self"])
    element = ctx.vars["object"]
    varname = ctx.visitor.varname_for_self_constraint(ctx.node)
    if isinstance(lst, SequenceIncompleteValue):
        no_return_unless = Constraint(
            varname,
            ConstraintType.is_value_object,
            True,
            SequenceIncompleteValue(list, (*lst.members, element)),
        )
        return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)
    elif isinstance(lst, GenericValue):
        expected_type = lst.get_generic_arg_for_type(list, ctx.visitor, 0)
        tv_map = expected_type.can_assign(element, ctx.visitor)
        if isinstance(tv_map, CanAssignError):
            ctx.show_error(
                f"Cannot append value of type {element} to list of {expected_type}",
                ErrorCode.incompatible_argument,
                arg="object",
                detail=str(tv_map),
            )
    return ImplReturn(KnownValue(None))


def _dict_setitem_impl(ctx: CallContext) -> ImplReturn:
    self_value = replace_known_sequence_value(ctx.vars["self"])
    key = ctx.vars["k"]
    value = ctx.vars["v"]
    # apparently for a[b] = c we get passed the AST node for a
    varname = ctx.visitor.varname_for_constraint(ctx.node)
    if isinstance(self_value, TypedDictValue):
        if not isinstance(key, KnownValue) or not isinstance(key.val, str):
            ctx.show_error(
                f"TypedDict key must be a string literal (got {key})",
                ErrorCode.invalid_typeddict_key,
                arg="k",
            )
        elif key.val not in self_value.items:
            ctx.show_error(
                f"Key {key.val!r} does not exist in {self_value}",
                ErrorCode.invalid_typeddict_key,
                arg="k",
            )
        else:
            expected_type = self_value.items[key.val]
            tv_map = expected_type.can_assign(value, ctx.visitor)
            if isinstance(tv_map, CanAssignError):
                ctx.show_error(
                    f"Value for key {key.val!r} must be {expected_type}, not {value}",
                    ErrorCode.incompatible_argument,
                    arg="v",
                    detail=str(tv_map),
                )
        return ImplReturn(KnownValue(None))
    elif isinstance(self_value, DictIncompleteValue):
        no_return_unless = Constraint(
            varname,
            ConstraintType.is_value_object,
            True,
            # This might create a duplicate but searching for that would
            # be O(n^2) and doesn't seem too useful.
            DictIncompleteValue([*self_value.items, (key, value)]),
        )
        return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)
    elif isinstance(self_value, TypedValue):
        key_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 0)
        tv_map = key_type.can_assign(key, ctx.visitor)
        if isinstance(tv_map, CanAssignError):
            ctx.show_error(
                f"Cannot set key of type {key} (expecting {key_type})",
                ErrorCode.incompatible_argument,
                arg="k",
                detail=str(tv_map),
            )
        value_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 1)
        tv_map = value_type.can_assign(value, ctx.visitor)
        if isinstance(tv_map, CanAssignError):
            ctx.show_error(
                f"Cannot set value of type {value} (expecting {value_type})",
                ErrorCode.incompatible_argument,
                arg="v",
                detail=str(tv_map),
            )
    return ImplReturn(KnownValue(None))


def _dict_getitem_impl(ctx: CallContext) -> ImplReturn:
    def inner(key: Value) -> Value:
        self_value = ctx.vars["self"]
        if isinstance(key, KnownValue):
            try:
                hash(key.val)
            except Exception:
                ctx.show_error(
                    f"Dictionary key {key} is not hashable",
                    ErrorCode.unhashable_key,
                    arg="k",
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
            if not TypedValue(str).is_assignable(key, ctx.visitor):
                ctx.show_error(
                    f"TypedDict key must be str, not {key}",
                    ErrorCode.invalid_typeddict_key,
                    arg="k",
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
                if dict_key.is_assignable(key, ctx.visitor)
            ]
            if not possible_values:
                # No error here, the key may have been added where we couldn't see it.
                return UNRESOLVED_VALUE
            return unite_values(*possible_values)
        elif isinstance(self_value, TypedValue):
            return self_value.get_generic_arg_for_type(dict, ctx.visitor, 1)
        else:
            return UNRESOLVED_VALUE

    return flatten_unions(inner, ctx.vars["k"])


def _list_add_impl(ctx: CallContext) -> ImplReturn:
    def inner(left: Value, right: Value) -> Value:
        left = replace_known_sequence_value(left)
        right = replace_known_sequence_value(right)
        if isinstance(left, SequenceIncompleteValue) and isinstance(
            right, SequenceIncompleteValue
        ):
            return SequenceIncompleteValue(list, [*left.members, *right.members])
        elif isinstance(left, TypedValue) and isinstance(right, TypedValue):
            left_args = left.get_generic_args_for_type(list, ctx.visitor)
            left_arg = left_args[0] if left_args else UNRESOLVED_VALUE
            right_args = right.get_generic_args_for_type(list, ctx.visitor)
            right_arg = right_args[0] if right_args else UNRESOLVED_VALUE
            return GenericValue(list, [unite_values(left_arg, right_arg)])
        else:
            return TypedValue(list)

    return flatten_unions(inner, ctx.vars["self"], ctx.vars["x"])


def _list_extend_impl(ctx: CallContext) -> ImplReturn:
    varname = ctx.visitor.varname_for_self_constraint(ctx.node)

    def inner(lst: Value, iterable: Value) -> ImplReturn:
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
                        collections.abc.Iterable, ctx.visitor, 0
                    )
                else:
                    arg_type = UNRESOLVED_VALUE
                constrained_value = GenericValue(
                    list, [unite_values(*lst.members, arg_type)]
                )
            no_return_unless = Constraint(
                varname, ConstraintType.is_value_object, True, constrained_value
            )
            return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)
        elif isinstance(lst, GenericValue):
            expected_type = lst.get_generic_arg_for_type(list, ctx.visitor, 0)
            if isinstance(iterable, TypedValue):
                actual_type = iterable.get_generic_arg_for_type(
                    collections.abc.Iterable, ctx.visitor, 0
                )
                tv_map = expected_type.can_assign(actual_type, ctx.visitor)
                if isinstance(tv_map, CanAssignError):
                    ctx.show_error(
                        f"Cannot extend list of {expected_type} with values of type"
                        f" {actual_type}",
                        ErrorCode.incompatible_argument,
                        arg="iterable",
                        detail=str(tv_map),
                    )
        return ImplReturn(KnownValue(None))

    return flatten_unions(inner, ctx.vars["self"], ctx.vars["iterable"])


def _set_add_impl(ctx: CallContext) -> ImplReturn:
    set_value = replace_known_sequence_value(ctx.vars["self"])
    element = ctx.vars["object"]
    varname = ctx.visitor.varname_for_self_constraint(ctx.node)
    if isinstance(set_value, SequenceIncompleteValue):
        no_return_unless = Constraint(
            varname,
            ConstraintType.is_value_object,
            True,
            SequenceIncompleteValue(set, (*set_value.members, element)),
        )
        return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)
    elif isinstance(set_value, GenericValue):
        set_args = set_value.get_generic_args_for_type(set, ctx.visitor)
        if set_args:
            expected_type = set_args[0]
            tv_map = expected_type.can_assign(element, ctx.visitor)
            if isinstance(tv_map, CanAssignError):
                ctx.show_error(
                    f"Cannot add value of type {element} to set of {expected_type}",
                    ErrorCode.incompatible_argument,
                    detail=str(tv_map),
                )
    return ImplReturn(KnownValue(None))


def _assert_is_value_impl(ctx: CallContext) -> Value:
    if not ctx.visitor._is_checking():
        return KnownValue(None)
    obj = ctx.vars["obj"]
    expected_value = ctx.vars["value"]
    if not isinstance(expected_value, KnownValue):
        ctx.show_error(
            "Value argument to assert_is_value must be a KnownValue (got"
            f" {expected_value}) {obj}",
            ErrorCode.inference_failure,
            arg="value",
        )
    else:
        if obj != expected_value.val:
            ctx.show_error(
                f"Bad value inference: expected {expected_value.val}, got {obj}",
                ErrorCode.inference_failure,
            )
    return KnownValue(None)


def _dump_value_impl(ctx: CallContext) -> Value:
    if ctx.visitor._is_checking():
        value = ctx.vars["value"]
        message = f"Value: {value}"
        if isinstance(value, KnownValue):
            sig = ctx.visitor.arg_spec_cache.get_argspec(value.val)
            if sig is not None:
                message += f", signature: {sig}"
        ctx.show_error(message, ErrorCode.inference_failure, arg="value")
    return KnownValue(None)


def _str_format_impl(ctx: CallContext) -> Value:
    self = ctx.vars["self"]
    if not isinstance(self, KnownValue):
        return TypedValue(str)
    args_value = replace_known_sequence_value(ctx.vars["args"])
    if not isinstance(args_value, SequenceIncompleteValue):
        return TypedValue(str)
    args = args_value.members
    kwargs_value = replace_known_sequence_value(ctx.vars["kwargs"])
    if not isinstance(kwargs_value, DictIncompleteValue):
        return TypedValue(str)
    kwargs = {}
    for key_value, value_value in kwargs_value.items:
        if isinstance(key_value, KnownValue) and isinstance(key_value.val, str):
            kwargs[key_value.val] = value_value
        else:
            return TypedValue(str)
    template = self.val
    used_indices = set()
    used_kwargs = set()
    current_index = 0
    parsed, errors = parse_format_string(template)
    if errors:
        _, message = errors[0]
        ctx.show_error(message, error_code=ErrorCode.incompatible_call)
        return TypedValue(str)
    for field in parsed.iter_replacement_fields():
        # TODO validate conversion specifiers, attributes, etc.
        if field.arg_name is None:
            if current_index >= len(args):
                ctx.show_error(
                    "Too few arguments to format string (expected at least"
                    f" {current_index})",
                    error_code=ErrorCode.incompatible_call,
                )
            used_indices.add(current_index)
            current_index += 1
        elif isinstance(field.arg_name, int):
            index = field.arg_name
            if index >= len(args):
                ctx.show_error(
                    f"Numbered argument {index} to format string is out of range",
                    error_code=ErrorCode.incompatible_call,
                )
            used_indices.add(index)
        else:
            if field.arg_name not in kwargs:
                ctx.show_error(
                    f"Named argument {field.arg_name} to format string was not given",
                    error_code=ErrorCode.incompatible_call,
                )
            used_kwargs.add(field.arg_name)
    unused_indices = set(range(len(args))) - used_indices
    if unused_indices:
        ctx.show_error(
            "Numbered argument(s) %s were not used"
            % ", ".join(map(str, sorted(unused_indices))),
            error_code=ErrorCode.incompatible_call,
        )
    unused_kwargs = set(kwargs) - used_kwargs
    if unused_kwargs:
        ctx.show_error(
            "Named argument(s) %s were not used" % ", ".join(sorted(unused_kwargs)),
            error_code=ErrorCode.incompatible_call,
        )
    return TypedValue(str)


def _cast_impl(ctx: CallContext) -> Value:
    typ = ctx.vars["typ"]
    return type_from_value(typ, visitor=ctx.visitor, node=ctx.node)


def _subclasses_impl(ctx: CallContext) -> Value:
    """Overridden because typeshed types make it (T) => List[T] instead."""
    self_obj = ctx.vars["self"]
    if isinstance(self_obj, KnownValue) and isinstance(self_obj.val, type):
        return KnownValue(self_obj.val.__subclasses__())
    return GenericValue(list, [TypedValue(type)])


def _assert_is_impl(ctx: CallContext) -> ImplReturn:
    return _qcore_assert_impl(ctx, ConstraintType.is_value, True)


def _assert_is_not_impl(ctx: CallContext) -> ImplReturn:
    return _qcore_assert_impl(ctx, ConstraintType.is_value, False)


def _qcore_assert_impl(
    ctx: CallContext, constraint_type: ConstraintType, positive: bool
) -> ImplReturn:
    left_varname = ctx.varname_for_arg("expected")
    right_varname = ctx.varname_for_arg("actual")
    if left_varname is not None and isinstance(ctx.vars["actual"], KnownValue):
        varname = left_varname
        constrained_to = ctx.vars["actual"].val
    elif right_varname is not None and isinstance(ctx.vars["expected"], KnownValue):
        varname = right_varname
        constrained_to = ctx.vars["expected"].val
    else:
        return ImplReturn(KnownValue(None))

    no_return_unless = Constraint(varname, constraint_type, positive, constrained_to)
    return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)


def len_of_value(val: Value) -> Value:
    if isinstance(val, SequenceIncompleteValue) and not issubclass(
        val.typ, KNOWN_MUTABLE_TYPES
    ):
        return KnownValue(len(val.members))
    elif isinstance(val, KnownValue):
        try:
            if not isinstance(val.val, KNOWN_MUTABLE_TYPES):
                return KnownValue(len(val.val))
        except Exception:
            return TypedValue(int)
    return TypedValue(int)


def _len_impl(ctx: CallContext) -> ImplReturn:
    varname = ctx.varname_for_arg("obj")
    if varname is None:
        constraint = NULL_CONSTRAINT
    else:
        constraint = PredicateProvider(varname, len_of_value)
    return ImplReturn(len_of_value(ctx.vars["obj"]), constraint)


_POS_ONLY = SigParameter.POSITIONAL_ONLY
_ENCODING_PARAMETER = SigParameter(
    "encoding", annotation=TypedValue(str), default=KnownValue("")
)


def get_default_argspecs() -> Dict[object, Signature]:
    signatures = [
        # pyanalyze helpers
        Signature.make(
            [SigParameter("obj"), SigParameter("value", annotation=TypedValue(Value))],
            impl=_assert_is_value_impl,
            callable=assert_is_value,
        ),
        Signature.make(
            [SigParameter("value")], impl=_dump_value_impl, callable=dump_value
        ),
        # builtins
        Signature.make(
            [SigParameter("self", _POS_ONLY)],
            callable=type.__subclasses__,
            impl=_subclasses_impl,
        ),
        Signature.make(
            [SigParameter("obj", _POS_ONLY), SigParameter("class_or_tuple", _POS_ONLY)],
            impl=_isinstance_impl,
            callable=isinstance,
        ),
        Signature.make(
            [
                SigParameter("cls", _POS_ONLY, annotation=TypedValue(type)),
                SigParameter(
                    "class_or_tuple",
                    _POS_ONLY,
                    annotation=MultiValuedValue(
                        [TypedValue(type), GenericValue(tuple, [TypedValue(type)])]
                    ),
                ),
            ],
            impl=_issubclass_impl,
            callable=issubclass,
        ),
        Signature.make(
            [
                SigParameter("object", _POS_ONLY),
                SigParameter("name", _POS_ONLY, annotation=TypedValue(str)),
                SigParameter("default", _POS_ONLY, default=_NO_ARG_SENTINEL),
            ],
            UNRESOLVED_VALUE,
            callable=getattr,
        ),
        Signature.make(
            [
                SigParameter("object", _POS_ONLY),
                SigParameter("name", _POS_ONLY, annotation=TypedValue(str)),
            ],
            impl=_hasattr_impl,
            callable=hasattr,
        ),
        Signature.make(
            [
                SigParameter("object", _POS_ONLY),
                SigParameter("name", _POS_ONLY, annotation=TypedValue(str)),
                SigParameter("value", _POS_ONLY),
            ],
            impl=_setattr_impl,
            callable=setattr,
        ),
        Signature.make(
            [
                SigParameter("type", _POS_ONLY, default=_NO_ARG_SENTINEL),
                SigParameter("obj", _POS_ONLY, default=_NO_ARG_SENTINEL),
            ],
            impl=_super_impl,
            callable=super,
        ),
        Signature.make(
            [SigParameter("iterable", _POS_ONLY, default=_NO_ARG_SENTINEL)],
            impl=_tuple_impl,
            callable=tuple,
        ),
        Signature.make(
            [SigParameter("iterable", _POS_ONLY, default=_NO_ARG_SENTINEL)],
            impl=_tuple_impl,
            callable=tuple,
        ),
        Signature.make(
            [SigParameter("iterable", _POS_ONLY, default=_NO_ARG_SENTINEL)],
            impl=_list_impl,
            callable=list,
        ),
        Signature.make(
            [SigParameter("iterable", _POS_ONLY, default=_NO_ARG_SENTINEL)],
            impl=_set_impl,
            callable=set,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(list)),
                SigParameter("object", _POS_ONLY),
            ],
            callable=list.append,
            impl=_list_append_impl,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(list)),
                SigParameter("x", _POS_ONLY, annotation=TypedValue(list)),
            ],
            callable=list.__add__,
            impl=_list_add_impl,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(list)),
                SigParameter(
                    "iterable",
                    _POS_ONLY,
                    annotation=TypedValue(collections.abc.Iterable),
                ),
            ],
            callable=list.extend,
            impl=_list_extend_impl,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(set)),
                SigParameter("object", _POS_ONLY),
            ],
            callable=set.add,
            impl=_set_add_impl,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(dict)),
                SigParameter("k", _POS_ONLY),
                SigParameter("v", _POS_ONLY),
            ],
            callable=dict.__setitem__,
            impl=_dict_setitem_impl,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(dict)),
                SigParameter("k", _POS_ONLY),
            ],
            callable=dict.__getitem__,
            impl=_dict_getitem_impl,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(bytes)),
                _ENCODING_PARAMETER,
                SigParameter(
                    "errors", annotation=TypedValue(str), default=KnownValue("")
                ),
            ],
            TypedValue(str),
            callable=bytes.decode,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(str)),
                _ENCODING_PARAMETER,
                SigParameter(
                    "errors", annotation=TypedValue(str), default=KnownValue("")
                ),
            ],
            TypedValue(bytes),
            callable=str.encode,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(str)),
                SigParameter("args", SigParameter.VAR_POSITIONAL),
                SigParameter("kwargs", SigParameter.VAR_KEYWORD),
            ],
            impl=_str_format_impl,
            callable=str.format,
        ),
        Signature.make(
            [SigParameter("typ"), SigParameter("val")], callable=cast, impl=_cast_impl
        ),
        # workaround for https://github.com/python/typeshed/pull/3501
        Signature.make(
            [
                SigParameter(
                    "message",
                    annotation=MultiValuedValue([TypedValue(str), TypedValue(Warning)]),
                ),
                SigParameter("category", default=KnownValue(None)),
                SigParameter(
                    "stacklevel", annotation=TypedValue(int), default=KnownValue(1)
                ),
            ],
            KnownValue(None),
            callable=warnings.warn,
        ),
        # just so we can infer the return value
        Signature.make([], NewTypeValue(qcore.Utime), callable=qcore.utime),
        Signature.make(
            [
                SigParameter("expected"),
                SigParameter("actual"),
                SigParameter("message", default=KnownValue(None)),
                SigParameter("extra", default=KnownValue(None)),
            ],
            callable=qcore.asserts.assert_is,
            impl=_assert_is_impl,
        ),
        Signature.make(
            [
                SigParameter("expected"),
                SigParameter("actual"),
                SigParameter("message", default=KnownValue(None)),
                SigParameter("extra", default=KnownValue(None)),
            ],
            callable=qcore.asserts.assert_is_not,
            impl=_assert_is_not_impl,
        ),
        Signature.make(
            [
                SigParameter("value"),
                SigParameter("types"),
                SigParameter("message", default=KnownValue(None)),
                SigParameter("extra", default=KnownValue(None)),
            ],
            callable=qcore.asserts.assert_is_instance,
            impl=_assert_is_instance_impl,
        ),
        # Need to override this because the type for the tp parameter in typeshed is too strict
        Signature.make(
            [SigParameter("name", annotation=TypedValue(str)), SigParameter(name="tp")],
            callable=NewType,
        ),
        Signature.make(
            [
                SigParameter(
                    "obj",
                    SigParameter.POSITIONAL_ONLY,
                    annotation=TypedValue(collections.abc.Sized),
                )
            ],
            callable=len,
            impl=_len_impl,
        ),
    ]
    return {sig.callable: sig for sig in signatures}
