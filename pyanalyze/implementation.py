import collections
import collections.abc
import inspect
import typing
from itertools import product
from typing import Callable, cast, Dict, NewType, Optional, Sequence, TypeVar, Union

import qcore
import typing_extensions

from .annotations import type_from_value
from .error_code import ErrorCode
from .extensions import assert_type, reveal_locals, reveal_type
from .format_strings import parse_format_string
from .predicates import IsAssignablePredicate
from .safe import hasattr_static, safe_isinstance, safe_issubclass
from .signature import (
    ANY_SIGNATURE,
    CallContext,
    ImplReturn,
    ParameterKind,
    Signature,
    SigParameter,
)
from .stacked_scopes import (
    AbstractConstraint,
    annotate_with_constraint,
    Composite,
    Constraint,
    ConstraintType,
    NULL_CONSTRAINT,
    OrConstraint,
    PredicateProvider,
    VarnameWithOrigin,
)
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    assert_is_value,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    check_hashability,
    concrete_values_from_iterable,
    DictIncompleteValue,
    dump_value,
    flatten_values,
    GenericValue,
    HasAttrGuardExtension,
    KNOWN_MUTABLE_TYPES,
    KnownValue,
    kv_pairs_from_mapping,
    KVPair,
    MultiValuedValue,
    NO_RETURN_VALUE,
    ParameterTypeGuardExtension,
    replace_known_sequence_value,
    SequenceValue,
    SubclassValue,
    TypedDictValue,
    TypedValue,
    TypeVarValue,
    unannotate,
    UNINITIALIZED_VALUE,
    unite_values,
    unpack_values,
    Value,
)

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
        list(flatten_values(val, unwrap_annotated=unwrap_annotated)) for val in values
    ]
    # If the lists are empty, we end up inferring Never as the return type, which
    # generally isn't right.
    value_lists = [lst if lst else [NO_RETURN_VALUE] for lst in value_lists]
    results = [
        clean_up_implementation_fn_return(callable(*vals))
        for vals in product(*value_lists)
    ]
    return ImplReturn.unite_impl_rets(results)


# Implementations of some important functions for use in their ExtendedArgSpecs (see above). These
# are called when the test_scope checker encounters call to these functions.


def _issubclass_impl(ctx: CallContext) -> Value:
    class_or_tuple = ctx.vars["class_or_tuple"]
    varname = ctx.varname_for_arg("cls")
    if varname is None or not isinstance(class_or_tuple, KnownValue):
        return TypedValue(bool)
    if isinstance(class_or_tuple.val, type):
        narrowed_type = SubclassValue(TypedValue(class_or_tuple.val))
    elif isinstance(class_or_tuple.val, tuple) and all(
        isinstance(elt, type) for elt in class_or_tuple.val
    ):
        vals = [SubclassValue(TypedValue(elt)) for elt in class_or_tuple.val]
        narrowed_type = unite_values(*vals)
    else:
        return TypedValue(bool)
    predicate = IsAssignablePredicate(narrowed_type, ctx.visitor, positive_only=False)
    constraint = Constraint(varname, ConstraintType.predicate, True, predicate)
    return annotate_with_constraint(TypedValue(bool), constraint)


def _isinstance_impl(ctx: CallContext) -> Value:
    class_or_tuple = ctx.vars["class_or_tuple"]
    varname = ctx.varname_for_arg("obj")
    if varname is None or not isinstance(class_or_tuple, KnownValue):
        return TypedValue(bool)
    if isinstance(class_or_tuple.val, type):
        narrowed_type = TypedValue(class_or_tuple.val)
    elif isinstance(class_or_tuple.val, tuple) and all(
        isinstance(elt, type) for elt in class_or_tuple.val
    ):
        vals = [TypedValue(elt) for elt in class_or_tuple.val]
        narrowed_type = unite_values(*vals)
    else:
        return TypedValue(bool)
    predicate = IsAssignablePredicate(narrowed_type, ctx.visitor, positive_only=False)
    constraint = Constraint(varname, ConstraintType.predicate, True, predicate)
    return annotate_with_constraint(TypedValue(bool), constraint)


def _constraint_from_isinstance(
    varname: Optional[VarnameWithOrigin], class_or_tuple: Value
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
        return OrConstraint.make(constraints)
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


def _record_attr_set(val: Value, name: str, ctx: CallContext) -> None:
    if isinstance(val, MultiValuedValue):
        for subval in val.vals:
            _record_attr_set(subval, name, ctx)
        return
    elif isinstance(val, AnnotatedValue):
        _record_attr_set(val.value, name, ctx)
        return
    elif isinstance(val, TypedValue):
        typ = val.typ
    elif isinstance(val, KnownValue):
        typ = type(val.val)
    else:
        return
    ctx.visitor._record_type_attr_set(
        typ, name, ctx.node, AnyValue(AnySource.inference)
    )


def _hasattr_impl(ctx: CallContext) -> Value:
    obj = ctx.vars["object"]
    name = ctx.vars["name"]
    if not isinstance(name, KnownValue) or not isinstance(name.val, str):
        return TypedValue(bool)
    # interpret a hasattr check as a sign that the object (somehow) has the attribute
    _record_attr_set(obj, name.val, ctx)

    # if the value exists on the type or instance, hasattr should return True
    # don't interpret the opposite to mean it should return False, as the attribute may
    # exist on a child class or get assigned at runtime
    if isinstance(obj, TypedValue) and obj.get_type_object(ctx.visitor).has_attribute(
        name.val, ctx.visitor
    ):
        return_value = KnownValue(True)
    elif isinstance(obj, KnownValue) and hasattr_static(obj.val, name.val):
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


def _tuple_impl(ctx: CallContext) -> ImplReturn:
    return _sequence_impl(tuple, ctx)


def _list_impl(ctx: CallContext) -> ImplReturn:
    return _sequence_impl(list, ctx)


def _set_impl(ctx: CallContext) -> ImplReturn:
    return _sequence_impl(set, ctx)


def _sequence_impl(typ: type, ctx: CallContext) -> ImplReturn:
    iterable = ctx.vars["iterable"]
    if iterable is _NO_ARG_SENTINEL:
        return ImplReturn(KnownValue(typ()))

    def inner(iterable: Value) -> Value:
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
            # TODO: Consider changing concrete_values_from_iterable to preserve unpacked bits
            return SequenceValue.make_or_known(typ, [(False, elt) for elt in cvi])

    return flatten_unions(inner, iterable)


def _list_append_impl(ctx: CallContext) -> ImplReturn:
    lst = replace_known_sequence_value(ctx.vars["self"])
    element = ctx.vars["object"]
    if isinstance(lst, SequenceValue):
        varname = ctx.visitor.varname_for_self_constraint(ctx.node)
        if varname is not None:
            no_return_unless = Constraint(
                varname,
                ConstraintType.is_value_object,
                True,
                SequenceValue.make_or_known(list, (*lst.members, (False, element))),
            )
            return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)
    if isinstance(lst, GenericValue):
        return _check_generic_container(
            "list.append", "object", ctx.vars["self"], lst, element, ctx, list
        )
    return ImplReturn(KnownValue(None))


def _sequence_getitem_impl(ctx: CallContext, typ: type) -> ImplReturn:
    def inner(key: Value) -> Value:
        self_value = replace_known_sequence_value(ctx.vars["self"])
        if not isinstance(self_value, TypedValue):
            return AnyValue(AnySource.error)  # shouldn't happen
        key = replace_known_sequence_value(key)
        if not TypedValue(slice).is_assignable(key, ctx.visitor):
            key, _ = ctx.visitor._check_dunder_call(
                ctx.ast_for_arg("obj"), Composite(key), "__index__", [], allow_call=True
            )

        if isinstance(key, KnownValue):
            if isinstance(key.val, int):
                if isinstance(self_value, SequenceValue):
                    members = self_value.get_member_sequence()
                    if members is not None:
                        if -len(members) <= key.val < len(members):
                            return members[key.val]
                        elif typ is list:
                            # fall back to the common type
                            return self_value.args[0]
                        else:
                            ctx.show_error(f"Tuple index out of range: {key}")
                            return AnyValue(AnySource.error)
                    else:
                        # The value contains at least one unpack. We try to find a precise
                        # type if everything leading up to the index we're interested in is
                        # a single element. For example, given a T: tuple[int, *tuple[str, ...]],
                        # T[0] should be int, but T[-1] should be int | str, because
                        # the unpacked tuple may be empty. For T[1] we could infer str, but
                        # we just infer int | str for simplicity.
                        if key.val >= 0:
                            for i, (is_many, member) in enumerate(self_value.members):
                                if is_many:
                                    # Give up
                                    break
                                if i == key.val:
                                    return member
                        else:
                            index_from_back = -key.val + 1
                            for i, (is_many, member) in enumerate(
                                reversed(self_value.members)
                            ):
                                if is_many:
                                    # Give up
                                    break
                                if i == index_from_back:
                                    return member
                    # fall back to the common type
                    return self_value.args[0]
                else:
                    return self_value.get_generic_arg_for_type(typ, ctx.visitor, 0)
            elif isinstance(key.val, slice):
                if isinstance(self_value, SequenceValue):
                    members = self_value.get_member_sequence()
                    if members is not None:
                        return SequenceValue.make_or_known(
                            typ, [(False, m) for m in members[key.val]]
                        )
                    else:
                        # If the value contains unpacked values, we don't attempt
                        # to resolve the slice.
                        return GenericValue(typ, self_value.args)
                elif self_value.typ in (list, tuple):
                    # For generics of exactly list/tuple, return the self type.
                    return self_value
                else:
                    # slicing a subclass of list or tuple returns a list
                    # or tuple, not a subclass (unless the subclass overrides
                    # __getitem__, but then we wouldn't get here).
                    # TODO return a more precise type if the class inherits
                    # from a generic list/tuple.
                    return TypedValue(typ)
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
        return
    if key.val not in self_value.items:
        if self_value.extra_keys is None:
            ctx.show_error(
                f"Key {key.val!r} does not exist in {self_value}",
                ErrorCode.invalid_typeddict_key,
                arg="k",
            )
            return
        else:
            expected_type = self_value.extra_keys
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


def _check_dict_key_hashability(key: Value, ctx: CallContext, arg: str) -> bool:
    hashability = check_hashability(key, ctx.visitor)
    if isinstance(hashability, CanAssignError):
        ctx.show_error(
            "Dictionary key is not hashable",
            ErrorCode.unhashable_key,
            arg=arg,
            detail=str(hashability),
        )
        return False
    return True


def _dict_setitem_impl(ctx: CallContext) -> ImplReturn:
    varname = ctx.varname_for_arg("self")
    key = ctx.vars["k"]
    if not _check_dict_key_hashability(key, ctx, "k"):
        return ImplReturn(KnownValue(None))
    pair = KVPair(key, ctx.vars["v"])
    return _add_pairs_to_dict(ctx.vars["self"], [pair], ctx, varname)


def _dict_getitem_impl(ctx: CallContext) -> ImplReturn:
    def inner(key: Value) -> Value:
        self_value = ctx.vars["self"]
        if isinstance(self_value, AnnotatedValue):
            self_value = self_value.value
        if not _check_dict_key_hashability(key, ctx, "k"):
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
                    if self_value.extra_keys is None:
                        ctx.show_error(
                            f"Unknown TypedDict key {key}",
                            ErrorCode.invalid_typeddict_key,
                            arg="k",
                        )
                        return AnyValue(AnySource.error)
            if self_value.extra_keys is not None:
                return self_value.extra_keys
            ctx.show_error(
                f"TypedDict key must be a literal, not {key}",
                ErrorCode.invalid_typeddict_key,
                arg="k",
            )
            return AnyValue(AnySource.error)
        elif isinstance(self_value, DictIncompleteValue):
            val = self_value.get_value(key, ctx.visitor)
            if val is UNINITIALIZED_VALUE:
                # No error here, the key may have been added where we couldn't see it.
                # TODO try out changing this
                return AnyValue(AnySource.error)
            return val
        elif isinstance(self_value, TypedValue):
            key_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 0)
            can_assign = key_type.can_assign(key, ctx.visitor)
            if isinstance(can_assign, CanAssignError):
                ctx.show_error(
                    f"Dictionary does not accept keys of type {key}",
                    error_code=ErrorCode.incompatible_argument,
                    detail=str(can_assign),
                    arg="key",
                )
            return self_value.get_generic_arg_for_type(dict, ctx.visitor, 1)
        else:
            return AnyValue(AnySource.inference)

    return flatten_unions(inner, ctx.vars["k"])


def _dict_get_impl(ctx: CallContext) -> ImplReturn:
    default = ctx.vars["default"]

    def inner(key: Value) -> Value:
        self_value = ctx.vars["self"]
        if isinstance(self_value, AnnotatedValue):
            self_value = self_value.value
        if not _check_dict_key_hashability(key, ctx, "k"):
            return AnyValue(AnySource.error)
        if isinstance(self_value, KnownValue):
            if isinstance(key, KnownValue):
                try:
                    return_value = self_value.val[key.val]
                except Exception:
                    return default
                else:
                    return KnownValue(return_value) | default
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
                    required, value = self_value.items[key.val]
                # probably KeyError, but catch anything in case it's an
                # unhashable str subclass or something
                except Exception:
                    if self_value.extra_keys is None:
                        ctx.show_error(
                            f"Unknown TypedDict key {key.val!r}",
                            ErrorCode.invalid_typeddict_key,
                            arg="k",
                        )
                        return AnyValue(AnySource.error)
                else:
                    if required:
                        return value
                    else:
                        return value | default
            if self_value.extra_keys is not None:
                return self_value.extra_keys | default
            ctx.show_error(
                f"TypedDict key must be a literal, not {key}",
                ErrorCode.invalid_typeddict_key,
                arg="k",
            )
            return AnyValue(AnySource.error)
        elif isinstance(self_value, DictIncompleteValue):
            val = self_value.get_value(key, ctx.visitor)
            if val is UNINITIALIZED_VALUE:
                return default
            return val | default
        elif isinstance(self_value, TypedValue):
            key_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 0)
            can_assign = key_type.can_assign(key, ctx.visitor)
            if isinstance(can_assign, CanAssignError):
                ctx.show_error(
                    f"Dictionary does not accept keys of type {key}",
                    error_code=ErrorCode.incompatible_argument,
                    detail=str(can_assign),
                    arg="key",
                )
            value_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 1)
            return value_type | default
        else:
            return AnyValue(AnySource.inference)

    return flatten_unions(inner, ctx.vars["key"])


def _dict_pop_impl(ctx: CallContext) -> ImplReturn:
    key = ctx.vars["key"]
    default = ctx.vars["default"]
    varname = ctx.visitor.varname_for_self_constraint(ctx.node)
    self_value = replace_known_sequence_value(ctx.vars["self"])

    if not _check_dict_key_hashability(key, ctx, "key"):
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
                is_required, expected_type = self_value.items[key.val]
            # probably KeyError, but catch anything in case it's an
            # unhashable str subclass or something
            except Exception:
                pass
            else:
                if is_required:
                    ctx.show_error(
                        f"Cannot pop required TypedDict key {key}",
                        error_code=ErrorCode.incompatible_argument,
                        arg="key",
                    )
                return ImplReturn(_maybe_unite(expected_type, default))
        if self_value.extra_keys is not None:
            return ImplReturn(_maybe_unite(self_value.extra_keys, default))
        ctx.show_error(
            f"Key {key} does not exist in TypedDict",
            ErrorCode.invalid_typeddict_key,
            arg="key",
        )
        return ImplReturn(default)
    elif isinstance(self_value, DictIncompleteValue):
        existing_value = self_value.get_value(key, ctx.visitor)
        is_present = existing_value is not UNINITIALIZED_VALUE
        if varname is not None and isinstance(key, KnownValue):
            new_value = DictIncompleteValue(
                self_value.typ,
                [pair for pair in self_value.kv_pairs if pair.key != key],
            )
            no_return_unless = Constraint(
                varname, ConstraintType.is_value_object, True, new_value
            )
        else:
            no_return_unless = NULL_CONSTRAINT
        if not is_present:
            if default is _NO_ARG_SENTINEL:
                ctx.show_error(
                    f"Key {key} does not exist in dictionary {self_value}",
                    error_code=ErrorCode.incompatible_argument,
                    arg="key",
                )
                return ImplReturn(AnyValue(AnySource.error))
            return ImplReturn(default, no_return_unless=no_return_unless)
        return ImplReturn(
            _maybe_unite(existing_value, default), no_return_unless=no_return_unless
        )
    elif isinstance(self_value, TypedValue):
        key_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 0)
        value_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 1)
        tv_map = key_type.can_assign(key, ctx.visitor)
        if isinstance(tv_map, CanAssignError):
            ctx.show_error(
                f"Key {key} is not valid for {self_value}",
                ErrorCode.incompatible_argument,
                arg="key",
            )
        return ImplReturn(_maybe_unite(value_type, default))
    else:
        return ImplReturn(AnyValue(AnySource.inference))


def _maybe_unite(value: Value, default: Value) -> Value:
    if default is _NO_ARG_SENTINEL:
        return value
    return unite_values(value, default)


def _dict_setdefault_impl(ctx: CallContext) -> ImplReturn:
    key = ctx.vars["key"]
    default = ctx.vars["default"]
    varname = ctx.visitor.varname_for_self_constraint(ctx.node)
    self_value = replace_known_sequence_value(ctx.vars["self"])

    if not _check_dict_key_hashability(key, ctx, "key"):
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
                        (
                            f"TypedDict key {key.val} expected value of type"
                            f" {expected_type}, not {default}"
                        ),
                        ErrorCode.incompatible_argument,
                        arg="default",
                    )
                return ImplReturn(expected_type)
        if self_value.extra_keys is not None:
            return ImplReturn(self_value.extra_keys | default)
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
        if varname is not None:
            no_return_unless = Constraint(
                varname, ConstraintType.is_value_object, True, new_value
            )
        else:
            no_return_unless = NULL_CONSTRAINT
        if not is_present:
            return ImplReturn(default, no_return_unless=no_return_unless)
        return ImplReturn(
            unite_values(default, existing_value), no_return_unless=no_return_unless
        )
    elif isinstance(self_value, TypedValue):
        key_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 0)
        value_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 1)
        new_value_type = unite_values(value_type, default)
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


def _update_incomplete_dict(
    self_val: Value,
    pairs: Sequence[KVPair],
    ctx: CallContext,
    varname: Optional[VarnameWithOrigin],
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
    varname: Optional[VarnameWithOrigin],
) -> ImplReturn:
    self_val = replace_known_sequence_value(self_val)
    if isinstance(self_val, TypedDictValue):
        for pair in pairs:
            _typeddict_setitem(self_val, pair.key, pair.value, ctx)
        return ImplReturn(KnownValue(None))
    elif isinstance(self_val, DictIncompleteValue):
        return _update_incomplete_dict(self_val, pairs, ctx, varname)
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
        return _update_incomplete_dict(self_val, pairs, ctx, varname)


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
        if isinstance(left, SequenceValue) and isinstance(right, SequenceValue):
            return SequenceValue.make_or_known(list, [*left.members, *right.members])
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
        if isinstance(cleaned_lst, SequenceValue):
            if isinstance(iterable, SequenceValue):
                constrained_value = SequenceValue.make_or_known(
                    list, (*cleaned_lst.members, *iterable.members)
                )
            else:
                if isinstance(iterable, TypedValue):
                    arg_type = iterable.get_generic_arg_for_type(
                        collections.abc.Iterable, ctx.visitor, 0
                    )
                else:
                    arg_type = AnyValue(AnySource.generic_argument)
                constrained_value = SequenceValue(
                    list, [*cleaned_lst.members, (True, arg_type)]
                )
            if return_container:
                return ImplReturn(constrained_value)
            if varname is not None:
                no_return_unless = Constraint(
                    varname, ConstraintType.is_value_object, True, constrained_value
                )
                return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)
        if isinstance(cleaned_lst, GenericValue) and isinstance(iterable, TypedValue):
            actual_type = iterable.get_generic_arg_for_type(
                collections.abc.Iterable, ctx.visitor, 0
            )
            return _check_generic_container(
                name,
                iterable_arg,
                lst,
                cleaned_lst,
                actual_type,
                ctx,
                list,
                return_container=return_container,
            )
        return ImplReturn(lst if return_container else KnownValue(None))

    return flatten_unions(inner, ctx.vars["self"], ctx.vars[iterable_arg])


def _list_extend_impl(ctx: CallContext) -> ImplReturn:
    return _list_extend_or_iadd_impl(ctx, "iterable", "list.extend")


def _list_iadd_impl(ctx: CallContext) -> ImplReturn:
    return _list_extend_or_iadd_impl(ctx, "x", "list.__iadd__", return_container=True)


def _check_generic_container(
    function_name: str,
    arg: str,
    original_container_type: Value,
    container_type: GenericValue,
    actual_type: Value,
    ctx: CallContext,
    typ: type,
    *,
    return_container: bool = False,
) -> ImplReturn:
    expected_type = container_type.get_generic_arg_for_type(typ, ctx.visitor, 0)
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
    if isinstance(set_value, SequenceValue):
        varname = ctx.visitor.varname_for_self_constraint(ctx.node)
        if varname is not None:
            no_return_unless = Constraint(
                varname,
                ConstraintType.is_value_object,
                True,
                SequenceValue.make_or_known(
                    set, (*set_value.members, (False, element))
                ),
            )
            return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)
    if isinstance(set_value, GenericValue):
        return _check_generic_container(
            "set.add", "object", ctx.vars["self"], set_value, element, ctx, set
        )
    return ImplReturn(KnownValue(None))


def _remove_annotated(val: Value) -> Value:
    if isinstance(val, AnnotatedValue):
        return _remove_annotated(val.value)
    elif isinstance(val, MultiValuedValue):
        return unite_values(*[_remove_annotated(subval) for subval in val.vals])
    return val


def _assert_is_value_impl(ctx: CallContext) -> Value:
    if not ctx.visitor._is_checking():
        return KnownValue(None)
    obj = ctx.vars["obj"]
    expected_value = ctx.vars["value"]
    if not isinstance(expected_value, KnownValue):
        ctx.show_error(
            (
                "Value argument to assert_is_value must be a KnownValue (got"
                f" {expected_value!r}; object is {obj!r})"
            ),
            ErrorCode.inference_failure,
            arg="value",
        )
    else:
        if _remove_annotated(ctx.vars["skip_annotated"]) == KnownValue(True):
            obj = _remove_annotated(obj)
        if obj != expected_value.val:
            ctx.show_error(
                f"Bad value inference: expected {expected_value.val}, got {obj}",
                ErrorCode.inference_failure,
            )
    return KnownValue(None)


def _reveal_type_impl(ctx: CallContext) -> Value:
    value = ctx.vars["value"]
    if ctx.visitor._is_checking():
        message = f"Revealed type is {ctx.visitor.display_value(value)}"
        ctx.show_error(message, ErrorCode.inference_failure, arg="value")
    return value


def _reveal_locals_impl(ctx: CallContext) -> Value:
    scope = ctx.visitor.scopes.current_scope()
    if ctx.visitor._is_collecting():
        for varname in scope.all_variables():
            scope.get(varname, ctx.node, ctx.visitor.state)
    else:
        details = []
        for varname in scope.all_variables():
            val, _, _ = scope.get(varname, ctx.node, ctx.visitor.state)
            details.append(CanAssignError(f"{varname}: {val}"))
        ctx.show_error(
            "Revealed local types are:",
            ErrorCode.inference_failure,
            detail=str(CanAssignError(children=details)),
        )
    return KnownValue(None)


def _dump_value_impl(ctx: CallContext) -> Value:
    value = ctx.vars["value"]
    if ctx.visitor._is_checking():
        message = f"Value is '{value!r}'"
        if isinstance(value, KnownValue):
            sig = ctx.visitor.arg_spec_cache.get_argspec(value.val)
            if sig is not None:
                message += f", signature is {sig!r}"
        ctx.show_error(message, ErrorCode.inference_failure, arg="value")
    return value


def _str_format_impl(ctx: CallContext) -> Value:
    self = ctx.vars["self"]
    if not isinstance(self, KnownValue):
        return TypedValue(str)
    args_value = replace_known_sequence_value(ctx.vars["args"])
    if isinstance(args_value, SequenceValue):
        args = args_value.get_member_sequence()
        if args is None:
            return TypedValue(str)
    else:
        return TypedValue(str)
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
                    (
                        "Too few arguments to format string (expected at least"
                        f" {current_index})"
                    ),
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


def _assert_type_impl(ctx: CallContext) -> Value:
    # TODO maybe we should walk over the whole value and remove Annotated.
    val = unannotate(ctx.vars["val"])
    typ = ctx.vars["typ"]
    expected_type = type_from_value(typ, visitor=ctx.visitor, node=ctx.node)
    if val != expected_type:
        ctx.show_error(
            f"Type is {val} (expected {expected_type})",
            error_code=ErrorCode.inference_failure,
            arg="obj",
        )
    return val


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
        isinstance(val, SequenceValue)
        and isinstance(val.typ, type)
        and not issubclass(val.typ, KNOWN_MUTABLE_TYPES)
    ):
        members = val.get_member_sequence()
        if members is not None:
            return KnownValue(len(members))
    if isinstance(val, KnownValue):
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


def _bool_impl(ctx: CallContext) -> Value:
    if ctx.vars["o"] is _NO_ARG_SENTINEL:
        return KnownValue(False)

    # Maybe we should check boolability here too? But it seems fair to
    # believe the author if they explicitly wrote bool().
    varname = ctx.varname_for_arg("o")
    if varname is None:
        return TypedValue(bool)
    constraint = Constraint(
        varname, ConstraintType.is_truthy, positive=True, value=None
    )
    return annotate_with_constraint(TypedValue(bool), constraint)


_POS_ONLY = ParameterKind.POSITIONAL_ONLY
_ENCODING_PARAMETER = SigParameter(
    "encoding", annotation=TypedValue(str), default=KnownValue("")
)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def get_default_argspecs() -> Dict[object, Signature]:
    signatures = [
        # pyanalyze helpers
        Signature.make(
            [
                SigParameter("obj"),
                SigParameter("value", annotation=TypedValue(Value)),
                SigParameter(
                    "skip_annotated",
                    ParameterKind.KEYWORD_ONLY,
                    default=KnownValue(False),
                    annotation=TypedValue(bool),
                ),
            ],
            KnownValue(None),
            impl=_assert_is_value_impl,
            callable=assert_is_value,
        ),
        Signature.make(
            [SigParameter("value", _POS_ONLY, annotation=TypeVarValue(T))],
            TypeVarValue(T),
            impl=_reveal_type_impl,
            callable=reveal_type,
        ),
        Signature.make(
            [], KnownValue(None), impl=_reveal_locals_impl, callable=reveal_locals
        ),
        Signature.make(
            [SigParameter("value", _POS_ONLY, annotation=TypeVarValue(T))],
            TypeVarValue(T),
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
            ],
            impl=_hasattr_impl,
            callable=hasattr_static,
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
            callable=dict.get,
            impl=_dict_get_impl,
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
                SigParameter("key", _POS_ONLY),
                SigParameter("default", _POS_ONLY, default=_NO_ARG_SENTINEL),
            ],
            callable=dict.pop,
            impl=_dict_pop_impl,
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(dict)),
                SigParameter("m", _POS_ONLY, default=_NO_ARG_SENTINEL),
                SigParameter("kwargs", ParameterKind.VAR_KEYWORD),
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
            DictIncompleteValue(
                dict, [KVPair(TypeVarValue(K), TypeVarValue(V), is_many=True)]
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
                SigParameter("args", ParameterKind.VAR_POSITIONAL),
                SigParameter("kwargs", ParameterKind.VAR_KEYWORD),
            ],
            impl=_str_format_impl,
            callable=str.format,
        ),
        Signature.make(
            [SigParameter("typ", _POS_ONLY), SigParameter("val", _POS_ONLY)],
            callable=cast,
            impl=_cast_impl,
        ),
        Signature.make(
            [
                SigParameter("val", _POS_ONLY, annotation=TypeVarValue(T)),
                SigParameter("typ", _POS_ONLY),
            ],
            TypeVarValue(T),
            callable=assert_type,
            impl=_assert_type_impl,
        ),
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
                    ParameterKind.POSITIONAL_ONLY,
                    annotation=TypedValue(collections.abc.Sized),
                )
            ],
            callable=len,
            impl=_len_impl,
        ),
        Signature.make(
            [
                SigParameter(
                    "o", ParameterKind.POSITIONAL_ONLY, default=_NO_ARG_SENTINEL
                )
            ],
            callable=bool,
            impl=_bool_impl,
        ),
        # Typeshed has it as TypeGuard[Callable[..., object]], which causes some
        # false positives.
        Signature.make(
            [
                SigParameter(
                    "obj", ParameterKind.POSITIONAL_ONLY, annotation=TypedValue(object)
                )
            ],
            callable=callable,
            return_annotation=AnnotatedValue(
                TypedValue(bool),
                [ParameterTypeGuardExtension("obj", CallableValue(ANY_SIGNATURE))],
            ),
        ),
    ]
    for mod in typing, typing_extensions:
        try:
            reveal_type_func = getattr(mod, "reveal_type")
        except AttributeError:
            pass
        else:
            # Anticipating https://bugs.python.org/issue46414
            sig = Signature.make(
                [SigParameter("value", _POS_ONLY, annotation=TypeVarValue(T))],
                TypeVarValue(T),
                impl=_reveal_type_impl,
                callable=reveal_type_func,
            )
            signatures.append(sig)
        # Anticipating that this will be added to the stdlib
        try:
            assert_type_func = getattr(mod, "assert_type")
        except AttributeError:
            pass
        else:
            sig = Signature.make(
                [
                    SigParameter("val", _POS_ONLY, annotation=TypeVarValue(T)),
                    SigParameter("typ", _POS_ONLY),
                ],
                TypeVarValue(T),
                callable=assert_type_func,
                impl=_assert_type_impl,
            )
            signatures.append(sig)
    return {sig.callable: sig for sig in signatures}
