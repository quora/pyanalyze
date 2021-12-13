from .annotations import type_from_value
from .error_code import ErrorCode
from .extensions import reveal_type
from .format_strings import parse_format_string
from .safe import safe_hasattr, safe_isinstance, safe_issubclass
from .stacked_scopes import (
    NULL_CONSTRAINT,
    AbstractConstraint,
    Composite,
    Constraint,
    ConstraintType,
    PredicateProvider,
    OrConstraint,
    Varname,
)
from .signature import ANY_SIGNATURE, SigParameter, Signature, ImplReturn, CallContext
from .value import (
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    HasAttrGuardExtension,
    KVPair,
    ParameterTypeGuardExtension,
    TypeVarValue,
    TypedValue,
    SubclassValue,
    GenericValue,
    NewTypeValue,
    DictIncompleteValue,
    SequenceIncompleteValue,
    TypedDictValue,
    KnownValue,
    MultiValuedValue,
    KNOWN_MUTABLE_TYPES,
    UnboundMethodValue,
    Value,
    WeakExtension,
    concrete_values_from_iterable,
    kv_pairs_from_mapping,
    make_weak,
    unite_values,
    flatten_values,
    replace_known_sequence_value,
    dump_value,
    assert_is_value,
    unpack_values,
)

from functools import reduce
import collections.abc
from itertools import product
import qcore
import inspect
import warnings
from types import FunctionType
from typing import Sequence, TypeVar, cast, Dict, NewType, Callable, Optional, Union

_NO_ARG_SENTINEL = KnownValue(qcore.MarkerObject("no argument given"))


def clean_up_implementation_fn_return(
    return_value: Union[Value, ImplReturn]
) -> ImplReturn:
    if isinstance(return_value, Value):
        return ImplReturn(return_value)
    return return_value


def flatten_unions(
    callable: Callable[..., Union[ImplReturn, Value]],
    *values: Value,
    unwrap_annotated: bool = False,
) -> ImplReturn:
    value_lists = [
        flatten_values(val, unwrap_annotated=unwrap_annotated) for val in values
    ]
    results = [
        clean_up_implementation_fn_return(callable(*vals))
        for vals in product(*value_lists)
    ]
    return ImplReturn.unite_impl_rets(results)


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
    for val in flatten_values(obj):
        if isinstance(val, (TypedValue)):
            typ = val.typ
        elif isinstance(val, KnownValue):
            typ = type(val.val)
        else:
            continue
        # interpret a hasattr check as a sign that the object (somehow) has the attribute
        ctx.visitor._record_type_attr_set(
            typ, name.val, ctx.node, AnyValue(AnySource.inference)
        )

    # if the value exists on the type or instance, hasattr should return True
    # don't interpret the opposite to mean it should return False, as the attribute may
    # exist on a child class or get assigned at runtime
    if isinstance(obj, TypedValue) and obj.get_type_object(ctx.visitor).has_attribute(
        name.val, ctx.visitor
    ):
        return_value = KnownValue(True)
    elif isinstance(obj, KnownValue) and safe_hasattr(obj.val, name.val):
        return_value = KnownValue(True)
    else:
        return_value = TypedValue(bool)
    metadata = [HasAttrGuardExtension("object", name, AnyValue(AnySource.inference))]
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
                return AnyValue(AnySource.error)
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
                    return AnyValue(AnySource.inference)
        return AnyValue(AnySource.inference)

    if isinstance(typ, KnownValue):
        if inspect.isclass(typ.val):
            cls = typ.val
        else:
            ctx.show_error(
                "First argument to super must be a class", ErrorCode.bad_super_call
            )
            return AnyValue(AnySource.error)
    else:
        return AnyValue(AnySource.inference)  # probably a dynamically created class

    if isinstance(obj, TypedValue) and obj.typ is not type:
        tobj = obj.get_type_object(ctx.visitor)
        is_value = True
    elif isinstance(obj, SubclassValue) and isinstance(obj.typ, TypedValue):
        tobj = obj.typ.get_type_object(ctx.visitor)
        is_value = False
    else:
        return AnyValue(AnySource.inference)

    if not tobj.is_assignable_to_type(cls):
        ctx.show_error("Incompatible arguments to super", ErrorCode.bad_super_call)

    current_class = ctx.visitor.asynq_checker.current_class
    if current_class is not None and cls is not current_class:
        ctx.show_error(
            "First argument to super() is not the current class",
            ErrorCode.bad_super_call,
        )

    if isinstance(tobj.typ, str):
        return AnyValue(AnySource.inference)

    try:
        super_val = super(cls, tobj.typ)
    except Exception:
        ctx.show_error("Bad arguments to super", ErrorCode.bad_super_call)
        return AnyValue(AnySource.error)

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
    cvi = concrete_values_from_iterable(iterable, ctx.visitor)
    if isinstance(cvi, CanAssignError):
        ctx.show_error(
            f"{iterable} is not iterable",
            ErrorCode.unsupported_operation,
            arg="iterable",
            detail=str(cvi),
        )
        return TypedValue(typ)
    elif isinstance(cvi, Value):
        return GenericValue(typ, [cvi])
    else:
        return SequenceIncompleteValue.make_or_known(typ, cvi)


def _list_append_impl(ctx: CallContext) -> ImplReturn:
    lst = replace_known_sequence_value(ctx.vars["self"])
    element = ctx.vars["object"]
    varname = ctx.visitor.varname_for_self_constraint(ctx.node)
    if isinstance(lst, SequenceIncompleteValue):
        no_return_unless = Constraint(
            varname,
            ConstraintType.is_value_object,
            True,
            SequenceIncompleteValue.make_or_known(list, (*lst.members, element)),
        )
        return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)
    elif isinstance(lst, GenericValue):
        return _maybe_broaden_weak_type(
            "list.append", "object", ctx.vars["self"], lst, element, ctx, list, varname
        )
    return ImplReturn(KnownValue(None))


def _sequence_getitem_impl(ctx: CallContext, typ: type) -> ImplReturn:
    def inner(key: Value) -> Value:
        self_value = replace_known_sequence_value(ctx.vars["self"])
        if not isinstance(self_value, TypedValue):
            return AnyValue(AnySource.error)  # shouldn't happen
        if not TypedValue(slice).is_assignable(key, ctx.visitor):
            key = ctx.visitor._check_dunder_call(
                ctx.ast_for_arg("obj"), Composite(key), "__index__", [], allow_call=True
            )

        if isinstance(key, KnownValue):
            if isinstance(key.val, int):
                if isinstance(self_value, SequenceIncompleteValue):
                    if -len(self_value.members) <= key.val < len(self_value.members):
                        return self_value.members[key.val]
                    elif typ is list:
                        # fall back to the common type
                        return self_value.args[0]
                    else:
                        ctx.show_error(f"Tuple index out of range: {key}")
                        return AnyValue(AnySource.error)
                else:
                    return self_value.get_generic_arg_for_type(typ, ctx.visitor, 0)
            elif isinstance(key.val, slice):
                if isinstance(self_value, SequenceIncompleteValue):
                    return SequenceIncompleteValue.make_or_known(
                        list, self_value.members[key.val]
                    )
                else:
                    return self_value
            else:
                ctx.show_error(f"Invalid {typ.__name__} key {key}")
                return AnyValue(AnySource.error)
        elif isinstance(key, TypedValue):
            tobj = key.get_type_object(ctx.visitor)
            if tobj.is_assignable_to_type(int):
                return self_value.get_generic_arg_for_type(typ, ctx.visitor, 0)
            elif tobj.is_assignable_to_type(slice):
                return self_value
            else:
                ctx.show_error(f"Invalid {typ.__name__} key {key}")
                return AnyValue(AnySource.error)
        elif isinstance(key, AnyValue):
            return AnyValue(AnySource.from_another)
        else:
            ctx.show_error(f"Invalid {typ.__name__} key {key}")
            return AnyValue(AnySource.error)

    return flatten_unions(inner, ctx.vars["obj"], unwrap_annotated=True)


def _list_getitem_impl(ctx: CallContext) -> ImplReturn:
    return _sequence_getitem_impl(ctx, list)


def _tuple_getitem_impl(ctx: CallContext) -> ImplReturn:
    return _sequence_getitem_impl(ctx, tuple)


def _typeddict_setitem(
    self_value: TypedDictValue, key: Value, value: Value, ctx: CallContext
) -> None:
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
        _, expected_type = self_value.items[key.val]
        tv_map = expected_type.can_assign(value, ctx.visitor)
        if isinstance(tv_map, CanAssignError):
            ctx.show_error(
                f"Value for key {key.val!r} must be {expected_type}, not {value}",
                ErrorCode.incompatible_argument,
                arg="v",
                detail=str(tv_map),
            )


def _dict_setitem_impl(ctx: CallContext) -> ImplReturn:
    varname = ctx.varname_for_arg("self")
    pair = KVPair(ctx.vars["k"], ctx.vars["v"])
    return _add_pairs_to_dict(ctx.vars["self"], [pair], ctx, varname)


def _dict_getitem_impl(ctx: CallContext) -> ImplReturn:
    def inner(key: Value) -> Value:
        self_value = ctx.vars["self"]
        if isinstance(self_value, AnnotatedValue):
            self_value = self_value.value
        if isinstance(key, KnownValue):
            try:
                hash(key.val)
            except Exception:
                ctx.show_error(
                    f"Dictionary key {key} is not hashable",
                    ErrorCode.unhashable_key,
                    arg="k",
                )
                return AnyValue(AnySource.error)
        if isinstance(self_value, KnownValue):
            if isinstance(key, KnownValue):
                try:
                    return_value = self_value.val[key.val]
                except Exception:
                    # No error here, the key may have been added where we couldn't see it.
                    return AnyValue(AnySource.error)
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
                return AnyValue(AnySource.error)
            elif isinstance(key, KnownValue):
                try:
                    _, value = self_value.items[key.val]
                    return value
                # probably KeyError, but catch anything in case it's an
                # unhashable str subclass or something
                except Exception:
                    # No error here; TypedDicts may have additional keys at runtime.
                    pass
            # TODO strictly we should throw an error for any non-Literal or unknown key:
            # https://www.python.org/dev/peps/pep-0589/#supported-and-unsupported-operations
            # Don't do that yet because it may cause too much disruption.
            return AnyValue(AnySource.inference)
        elif isinstance(self_value, DictIncompleteValue):
            val = self_value.get_value(key, ctx.visitor)
            if val is UNINITIALIZED_VALUE:
                # No error here, the key may have been added where we couldn't see it.
                # TODO try out changing this
                return AnyValue(AnySource.error)
            return val
        elif isinstance(self_value, TypedValue):
            return self_value.get_generic_arg_for_type(dict, ctx.visitor, 1)
        else:
            return AnyValue(AnySource.inference)

    return flatten_unions(inner, ctx.vars["k"])


def _dict_setdefault_impl(ctx: CallContext) -> ImplReturn:
    key = ctx.vars["key"]
    default = ctx.vars["default"]
    varname = ctx.visitor.varname_for_self_constraint(ctx.node)
    self_value = replace_known_sequence_value(ctx.vars["self"])

    if isinstance(key, KnownValue):
        try:
            hash(key.val)
        except Exception:
            ctx.show_error(
                f"Dictionary key {key} is not hashable",
                ErrorCode.unhashable_key,
                arg="key",
            )
            return ImplReturn(AnyValue(AnySource.error))

    if isinstance(self_value, TypedDictValue):
        if not TypedValue(str).is_assignable(key, ctx.visitor):
            ctx.show_error(
                f"TypedDict key must be str, not {key}",
                ErrorCode.invalid_typeddict_key,
                arg="key",
            )
            return ImplReturn(AnyValue(AnySource.error))
        elif isinstance(key, KnownValue):
            try:
                _, expected_type = self_value.items[key.val]
            # probably KeyError, but catch anything in case it's an
            # unhashable str subclass or something
            except Exception:
                pass
            else:
                tv_map = expected_type.can_assign(default, ctx.visitor)
                if isinstance(tv_map, CanAssignError):
                    ctx.show_error(
                        f"TypedDict key {key.val} expected value of type"
                        f" {expected_type}, not {default}",
                        ErrorCode.incompatible_argument,
                        arg="default",
                    )
                return ImplReturn(expected_type)
        ctx.show_error(
            f"Key {key} does not exist in TypedDict",
            ErrorCode.invalid_typeddict_key,
            arg="key",
        )
        return ImplReturn(default)
    elif isinstance(self_value, DictIncompleteValue):
        existing_value = self_value.get_value(key, ctx.visitor)
        is_present = existing_value is not UNINITIALIZED_VALUE
        new_value = DictIncompleteValue(
            self_value.typ,
            [*self_value.kv_pairs, KVPair(key, default, is_required=not is_present)],
        )
        no_return_unless = Constraint(
            varname, ConstraintType.is_value_object, True, new_value
        )
        if not is_present:
            return ImplReturn(default, no_return_unless=no_return_unless)
        return ImplReturn(
            unite_values(default, existing_value), no_return_unless=no_return_unless
        )
    elif isinstance(self_value, TypedValue):
        key_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 0)
        value_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 1)
        new_value_type = unite_values(value_type, default)
        if _is_weak(ctx.vars["self"]):
            new_key_type = unite_values(key_type, key)
            new_type = make_weak(
                GenericValue(self_value.typ, [new_key_type, new_value_type])
            )
            no_return_unless = Constraint(
                varname, ConstraintType.is_value_object, True, new_type
            )
            return ImplReturn(new_value_type, no_return_unless=no_return_unless)
        else:
            tv_map = key_type.can_assign(key, ctx.visitor)
            if isinstance(tv_map, CanAssignError):
                ctx.show_error(
                    f"Key {key} is not valid for {self_value}",
                    ErrorCode.incompatible_argument,
                    arg="key",
                )
            return ImplReturn(new_value_type)
    else:
        return ImplReturn(AnyValue(AnySource.inference))


def _unpack_iterable_of_pairs(
    val: Value, ctx: CanAssignContext
) -> Union[Sequence[KVPair], CanAssignError]:
    concrete = concrete_values_from_iterable(val, ctx)
    if isinstance(concrete, CanAssignError):
        return concrete
    if isinstance(concrete, Value):
        vals = unpack_values(concrete, ctx, 2)
        if isinstance(vals, CanAssignError):
            return CanAssignError(f"{concrete} is not a key-value pair", [vals])
        return [KVPair(vals[0], vals[1], is_many=True)]
    kv_pairs = []
    for i, subval in enumerate(concrete):
        vals = unpack_values(subval, ctx, 2)
        if isinstance(vals, CanAssignError):
            child = CanAssignError(f"{concrete} is not a key-value pair", [vals])
            return CanAssignError(f"In member {i} of iterable {val}", [child])
        kv_pairs.append(KVPair(vals[0], vals[1]))
    return kv_pairs


def _weak_dict_update(
    self_val: Value,
    pairs: Sequence[KVPair],
    ctx: CallContext,
    varname: Optional[Varname],
) -> ImplReturn:
    self_pairs = kv_pairs_from_mapping(self_val, ctx.visitor)
    if isinstance(self_pairs, CanAssignError):
        ctx.show_error("self is not a mapping", arg="self", detail=str(self_pairs))
        return ImplReturn(KnownValue(None))
    pairs = [*self_pairs, *pairs]

    if varname is not None:
        no_return_unless = Constraint(
            varname,
            ConstraintType.is_value_object,
            True,
            DictIncompleteValue(
                self_val.typ if isinstance(self_val, TypedValue) else dict, pairs
            ),
        )
        return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)

    return ImplReturn(KnownValue(None))


def _add_pairs_to_dict(
    self_val: Value,
    pairs: Sequence[KVPair],
    ctx: CallContext,
    varname: Optional[Varname],
) -> ImplReturn:
    if _is_weak(self_val):
        return _weak_dict_update(self_val, pairs, ctx, varname)

    # Now we don't care about Annotated
    self_val = replace_known_sequence_value(self_val)
    if isinstance(self_val, TypedDictValue):
        for pair in pairs:
            _typeddict_setitem(self_val, pair.key, pair.value, ctx)
        return ImplReturn(KnownValue(None))
    elif isinstance(self_val, DictIncompleteValue):
        return _weak_dict_update(self_val, pairs, ctx, varname)
    elif isinstance(self_val, TypedValue):
        key_type = self_val.get_generic_arg_for_type(dict, ctx.visitor, 0)
        value_type = self_val.get_generic_arg_for_type(dict, ctx.visitor, 1)
        for pair in pairs:
            tv_map = key_type.can_assign(pair.key, ctx.visitor)
            if isinstance(tv_map, CanAssignError):
                ctx.show_error(
                    f"Cannot set key of type {pair.key} (expecting {key_type})",
                    ErrorCode.incompatible_argument,
                    arg="k",
                    detail=str(tv_map),
                )
            tv_map = value_type.can_assign(pair.value, ctx.visitor)
            if isinstance(tv_map, CanAssignError):
                ctx.show_error(
                    f"Cannot set value of type {pair.value} (expecting {value_type})",
                    ErrorCode.incompatible_argument,
                    arg="v",
                    detail=str(tv_map),
                )
        return ImplReturn(KnownValue(None))
    else:
        return _weak_dict_update(self_val, pairs, ctx, varname)


def _dict_update_impl(ctx: CallContext) -> ImplReturn:
    def inner(self_val: Value, m_val: Value, kwargs_val: Value) -> ImplReturn:
        pairs = []
        # The second argument must be either a mapping or an iterable of key-value pairs.
        if m_val is not _NO_ARG_SENTINEL:
            m_pairs = kv_pairs_from_mapping(m_val, ctx.visitor)
            if isinstance(m_pairs, CanAssignError):
                # Try an iterable of pairs instead
                iterable_pairs = _unpack_iterable_of_pairs(m_val, ctx.visitor)
                if isinstance(iterable_pairs, CanAssignError):
                    error = CanAssignError(children=[m_pairs, iterable_pairs])
                    ctx.show_error(
                        "m is not a mapping or iterable", arg="self", detail=str(error)
                    )
                else:
                    pairs += iterable_pairs
            else:
                pairs += m_pairs

        # Separate **kwargs
        kwargs_pairs = kv_pairs_from_mapping(kwargs_val, ctx.visitor)
        if isinstance(kwargs_pairs, CanAssignError):
            # should never happen
            ctx.show_error(
                "kwargs is not a mapping", arg="kwargs", detail=str(kwargs_pairs)
            )
            return ImplReturn(KnownValue(None))
        pairs += kwargs_pairs

        varname = ctx.visitor.varname_for_self_constraint(ctx.node)
        return _add_pairs_to_dict(self_val, pairs, ctx, varname)

    return flatten_unions(inner, ctx.vars["self"], ctx.vars["m"], ctx.vars["kwargs"])


def _dict_keys_impl(ctx: CallContext) -> Value:
    self_value = replace_known_sequence_value(ctx.vars["self"])
    if not isinstance(self_value, TypedValue):
        return TypedValue(collections.abc.KeysView)
    key_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 0)
    return GenericValue(collections.abc.KeysView, [key_type])


def _dict_items_impl(ctx: CallContext) -> Value:
    self_value = replace_known_sequence_value(ctx.vars["self"])
    if not isinstance(self_value, TypedValue):
        return TypedValue(collections.abc.ItemsView)
    key_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 0)
    value_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 1)
    return GenericValue(collections.abc.ItemsView, [key_type, value_type])


def _dict_values_impl(ctx: CallContext) -> Value:
    self_value = replace_known_sequence_value(ctx.vars["self"])
    if not isinstance(self_value, TypedValue):
        return TypedValue(collections.abc.ValuesView)
    value_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 1)
    return GenericValue(collections.abc.ValuesView, [value_type])


def _list_add_impl(ctx: CallContext) -> ImplReturn:
    def inner(left: Value, right: Value) -> Value:
        left = replace_known_sequence_value(left)
        right = replace_known_sequence_value(right)
        if isinstance(left, SequenceIncompleteValue) and isinstance(
            right, SequenceIncompleteValue
        ):
            return SequenceIncompleteValue.make_or_known(
                list, [*left.members, *right.members]
            )
        elif isinstance(left, TypedValue) and isinstance(right, TypedValue):
            left_arg = left.get_generic_arg_for_type(list, ctx.visitor, 0)
            right_arg = right.get_generic_arg_for_type(list, ctx.visitor, 0)
            return GenericValue(list, [unite_values(left_arg, right_arg)])
        else:
            return TypedValue(list)

    return flatten_unions(inner, ctx.vars["self"], ctx.vars["x"])


def _list_extend_or_iadd_impl(
    ctx: CallContext, iterable_arg: str, name: str, *, return_container: bool = False
) -> ImplReturn:
    varname = ctx.visitor.varname_for_self_constraint(ctx.node)

    def inner(lst: Value, iterable: Value) -> ImplReturn:
        cleaned_lst = replace_known_sequence_value(lst)
        iterable = replace_known_sequence_value(iterable)
        if isinstance(cleaned_lst, SequenceIncompleteValue):
            if isinstance(
                iterable, SequenceIncompleteValue
            ) and iterable.get_type_object(ctx.visitor).is_exactly((list, tuple)):
                constrained_value = SequenceIncompleteValue.make_or_known(
                    list, (*cleaned_lst.members, *iterable.members)
                )
            else:
                if isinstance(iterable, TypedValue):
                    arg_type = iterable.get_generic_arg_for_type(
                        collections.abc.Iterable, ctx.visitor, 0
                    )
                else:
                    arg_type = AnyValue(AnySource.generic_argument)
                generic_arg = unite_values(*cleaned_lst.members, arg_type)
                constrained_value = make_weak(GenericValue(list, [generic_arg]))
            if return_container:
                return ImplReturn(constrained_value)
            no_return_unless = Constraint(
                varname, ConstraintType.is_value_object, True, constrained_value
            )
            return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)
        elif isinstance(cleaned_lst, GenericValue) and isinstance(iterable, TypedValue):
            actual_type = iterable.get_generic_arg_for_type(
                collections.abc.Iterable, ctx.visitor, 0
            )
            return _maybe_broaden_weak_type(
                name,
                iterable_arg,
                lst,
                cleaned_lst,
                actual_type,
                ctx,
                list,
                varname,
                return_container=return_container,
            )
        return ImplReturn(lst if return_container else KnownValue(None))

    return flatten_unions(inner, ctx.vars["self"], ctx.vars[iterable_arg])


def _list_extend_impl(ctx: CallContext) -> ImplReturn:
    return _list_extend_or_iadd_impl(ctx, "iterable", "list.extend")


def _list_iadd_impl(ctx: CallContext) -> ImplReturn:
    return _list_extend_or_iadd_impl(ctx, "x", "list.__iadd__", return_container=True)


def _is_weak(val: Value) -> bool:
    return isinstance(val, AnnotatedValue) and val.has_metadata_of_type(WeakExtension)


def _maybe_broaden_weak_type(
    function_name: str,
    arg: str,
    original_container_type: Value,
    container_type: Value,
    actual_type: Value,
    ctx: CallContext,
    typ: type,
    varname: Varname,
    *,
    return_container: bool = False,
) -> ImplReturn:
    expected_type = container_type.get_generic_arg_for_type(typ, ctx.visitor, 0)
    if _is_weak(original_container_type):
        generic_arg = unite_values(expected_type, actual_type)
        constrained_value = make_weak(GenericValue(typ, [generic_arg]))
        no_return_unless = Constraint(
            varname, ConstraintType.is_value_object, True, constrained_value
        )
        if return_container:
            return ImplReturn(constrained_value)
        return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)

    tv_map = expected_type.can_assign(actual_type, ctx.visitor)
    if isinstance(tv_map, CanAssignError):
        ctx.show_error(
            f"{function_name}: expected {expected_type} but got {actual_type}",
            ErrorCode.incompatible_argument,
            arg=arg,
            detail=str(tv_map),
        )
    if return_container:
        return ImplReturn(original_container_type)
    return ImplReturn(KnownValue(None))


def _set_add_impl(ctx: CallContext) -> ImplReturn:
    set_value = replace_known_sequence_value(ctx.vars["self"])
    element = ctx.vars["object"]
    varname = ctx.visitor.varname_for_self_constraint(ctx.node)
    if isinstance(set_value, SequenceIncompleteValue):
        no_return_unless = Constraint(
            varname,
            ConstraintType.is_value_object,
            True,
            SequenceIncompleteValue.make_or_known(set, (*set_value.members, element)),
        )
        return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)
    elif isinstance(set_value, GenericValue):
        return _maybe_broaden_weak_type(
            "set.add", "object", ctx.vars["self"], set_value, element, ctx, set, varname
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
            f" {expected_value!r}; object is {obj!r})",
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


def _reveal_type_impl(ctx: CallContext) -> Value:
    if ctx.visitor._is_checking():
        value = ctx.vars["value"]
        message = f"Revealed type is '{value!s}'"
        if isinstance(value, KnownValue):
            sig = ctx.visitor.arg_spec_cache.get_argspec(value.val)
        elif isinstance(value, UnboundMethodValue):
            sig = value.get_signature(ctx.visitor)
        else:
            sig = None
        if sig is not None:
            message += f", signature is {sig!s}"
        ctx.show_error(message, ErrorCode.inference_failure, arg="value")
    return KnownValue(None)


def _dump_value_impl(ctx: CallContext) -> Value:
    if ctx.visitor._is_checking():
        value = ctx.vars["value"]
        message = f"Value is '{value!r}'"
        if isinstance(value, KnownValue):
            sig = ctx.visitor.arg_spec_cache.get_argspec(value.val)
            if sig is not None:
                message += f", signature is {sig!r}"
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
    kwargs = {}
    if isinstance(kwargs_value, DictIncompleteValue):
        for pair in kwargs_value.kv_pairs:
            if isinstance(pair.key, KnownValue) and isinstance(pair.key.val, str):
                kwargs[pair.key.val] = pair.value
            else:
                return TypedValue(str)
    elif isinstance(kwargs_value, TypedDictValue):
        for key, (required, value_value) in kwargs_value.items.items():
            if required:
                kwargs[key] = value_value
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
    # Skip these checks in unions because the arguments may be used in a
    # different branch of the union. Ideally we'd error if they are unused
    # in all variants, but that's difficult to achieve with the current
    # abstractions.
    if not ctx.visitor.in_union_decomposition:
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
    if (
        isinstance(val, SequenceIncompleteValue)
        and isinstance(val.typ, type)
        and not issubclass(val.typ, KNOWN_MUTABLE_TYPES)
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

K = TypeVar("K")
V = TypeVar("V")


def get_default_argspecs() -> Dict[object, Signature]:
    signatures = [
        # pyanalyze helpers
        Signature.make(
            [SigParameter("obj"), SigParameter("value", annotation=TypedValue(Value))],
            KnownValue(None),
            impl=_assert_is_value_impl,
            callable=assert_is_value,
        ),
        Signature.make(
            [SigParameter("value", _POS_ONLY)],
            KnownValue(None),
            impl=_reveal_type_impl,
            callable=reveal_type,
        ),
        Signature.make(
            [SigParameter("value", _POS_ONLY)],
            KnownValue(None),
            impl=_dump_value_impl,
            callable=dump_value,
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
            [SigParameter("obj"), SigParameter("class_or_tuple")],
            impl=_isinstance_impl,
            callable=safe_isinstance,
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
            callable=safe_issubclass,
        ),
        Signature.make(
            [
                SigParameter("object", _POS_ONLY),
                SigParameter("name", _POS_ONLY, annotation=TypedValue(str)),
                SigParameter("default", _POS_ONLY, default=_NO_ARG_SENTINEL),
            ],
            AnyValue(AnySource.inference),
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
                    "x", _POS_ONLY, annotation=TypedValue(collections.abc.Iterable)
                ),
            ],
            callable=list.__iadd__,
            impl=_list_iadd_impl,
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
                SigParameter("self", _POS_ONLY, annotation=TypedValue(list)),
                SigParameter("obj", _POS_ONLY),
            ],
            callable=list.__getitem__,
            impl=_list_getitem_impl,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(tuple)),
                SigParameter("obj", _POS_ONLY),
            ],
            callable=tuple.__getitem__,
            impl=_tuple_getitem_impl,
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
                SigParameter("self", _POS_ONLY, annotation=TypedValue(dict)),
                SigParameter("key", _POS_ONLY),
                SigParameter("default", _POS_ONLY, default=KnownValue(None)),
            ],
            callable=dict.setdefault,
            impl=_dict_setdefault_impl,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(dict)),
                SigParameter("m", _POS_ONLY, default=_NO_ARG_SENTINEL),
                SigParameter("kwargs", SigParameter.VAR_KEYWORD),
            ],
            KnownValue(None),
            callable=dict.update,
            impl=_dict_update_impl,
        ),
        Signature.make(
            [
                SigParameter(
                    "self",
                    _POS_ONLY,
                    annotation=GenericValue(dict, [TypeVarValue(K), TypeVarValue(V)]),
                )
            ],
            AnnotatedValue(
                GenericValue(dict, [TypeVarValue(K), TypeVarValue(V)]),
                [WeakExtension()],
            ),
            callable=dict.copy,
        ),
        # Implementations of keys/items/values to compensate for incomplete
        # typeshed support. In the stubs these return instances of a private class
        # that doesn't exist in reality.
        Signature.make(
            [SigParameter("self", _POS_ONLY, annotation=TypedValue(dict))],
            callable=dict.keys,
            impl=_dict_keys_impl,
        ),
        Signature.make(
            [SigParameter("self", _POS_ONLY, annotation=TypedValue(dict))],
            callable=dict.values,
            impl=_dict_values_impl,
        ),
        Signature.make(
            [SigParameter("self", _POS_ONLY, annotation=TypedValue(dict))],
            callable=dict.items,
            impl=_dict_items_impl,
        ),
        Signature.make(
            [SigParameter("self", _POS_ONLY, annotation=TypedValue(dict))],
            callable=collections.OrderedDict.keys,
            impl=_dict_keys_impl,
        ),
        Signature.make(
            [SigParameter("self", _POS_ONLY, annotation=TypedValue(dict))],
            callable=collections.OrderedDict.values,
            impl=_dict_values_impl,
        ),
        Signature.make(
            [SigParameter("self", _POS_ONLY, annotation=TypedValue(dict))],
            callable=collections.OrderedDict.items,
            impl=_dict_items_impl,
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
            allow_call=True,
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
            allow_call=True,
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
        # The overloaded annotation in typeshed causes a couple of problems:
        # - sorted(Sequence[A] | Sequence[B]) turns into List[A | B] instead of List[A] | List[B]
        #   pyright agrees with pyanalyze here but mypy somehow infers the second type.
        # - The bounded TypeVar on the key argument makes pyanalyze infer the
        #   return type as list[Sized] if we use key=len. Fixing this may require changing the
        #   TypeVar resolution algorithm.
        # To avoid these problems, we use a more permissive hardcoded signature for now.
        Signature.make(
            [
                SigParameter(
                    "iterable",
                    SigParameter.POSITIONAL_ONLY,
                    annotation=TypedValue(collections.abc.Iterable),
                ),
                SigParameter(
                    "key",
                    SigParameter.KEYWORD_ONLY,
                    annotation=CallableValue(
                        Signature.make(
                            [SigParameter("arg", SigParameter.POSITIONAL_ONLY)],
                            return_annotation=TypedValue("_typeshed.SupportsLessThan"),
                        )
                    ),
                    default=KnownValue(None),
                ),
                SigParameter(
                    "reverse",
                    SigParameter.KEYWORD_ONLY,
                    annotation=TypedValue(bool),
                    default=KnownValue(False),
                ),
            ],
            return_annotation=TypedValue(list),
            allow_call=True,
            callable=sorted,
        ),
        # TypeGuards, which aren't in typeshed yet
        Signature.make(
            [
                SigParameter(
                    "obj", SigParameter.POSITIONAL_ONLY, annotation=TypedValue(object)
                )
            ],
            callable=callable,
            return_annotation=AnnotatedValue(
                TypedValue(bool),
                [ParameterTypeGuardExtension("obj", CallableValue(ANY_SIGNATURE))],
            ),
        ),
        Signature.make(
            [
                SigParameter(
                    "object",
                    SigParameter.POSITIONAL_OR_KEYWORD,
                    annotation=TypedValue(object),
                )
            ],
            callable=inspect.isfunction,
            return_annotation=AnnotatedValue(
                TypedValue(bool),
                [ParameterTypeGuardExtension("object", TypedValue(FunctionType))],
            ),
        ),
    ]
    return {sig.callable: sig for sig in signatures}
