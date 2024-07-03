import ast
import collections
import collections.abc
import inspect
import typing
from itertools import product
from typing import (
    Callable,
    Dict,
    Iterable,
    NewType,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import qcore
import typing_extensions

from . import runtime
from .annotated_types import MaxLen, MinLen
from .annotations import type_from_value
from .error_code import ErrorCode
from .extensions import assert_type, reveal_locals, reveal_type
from .format_strings import parse_format_string
from .predicates import IsAssignablePredicate
from .safe import hasattr_static, is_union, safe_isinstance, safe_issubclass
from .signature import (
    ANY_SIGNATURE,
    CallContext,
    ImplReturn,
    ParameterKind,
    Signature,
    SigParameter,
)
from .stacked_scopes import (
    NULL_CONSTRAINT,
    AbstractConstraint,
    Composite,
    Constraint,
    ConstraintType,
    OrConstraint,
    PredicateProvider,
    VarnameWithOrigin,
    annotate_with_constraint,
)
from .value import (
    KNOWN_MUTABLE_TYPES,
    NO_RETURN_VALUE,
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    CustomCheckExtension,
    DictIncompleteValue,
    GenericValue,
    HasAttrGuardExtension,
    KnownValue,
    KVPair,
    MultiValuedValue,
    ParameterTypeGuardExtension,
    SequenceValue,
    SubclassValue,
    TypedDictValue,
    TypedValue,
    TypeVarValue,
    Value,
    annotate_value,
    assert_is_value,
    check_hashability,
    concrete_values_from_iterable,
    dump_value,
    flatten_values,
    kv_pairs_from_mapping,
    replace_known_sequence_value,
    unite_values,
    unpack_values,
)

_NO_ARG_SENTINEL = KnownValue(qcore.MarkerObject("no argument given"))


def clean_up_implementation_fn_return(
    return_value: Union[Value, ImplReturn],
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
    try:
        narrowed_types = list(_resolve_isinstance_arg(class_or_tuple.val))
    except _CannotResolve as e:
        ctx.show_error(
            f'Second argument to "issubclass" must be a type, union,'
            f' or tuple of types, not "{e.args[0]!r}"',
            ErrorCode.incompatible_argument,
            arg="class_or_tuple",
        )
        return TypedValue(bool)
    narrowed_type = unite_values(
        *[SubclassValue(TypedValue(typ)) for typ in narrowed_types]
    )
    predicate = IsAssignablePredicate(narrowed_type, ctx.visitor, positive_only=False)
    constraint = Constraint(varname, ConstraintType.predicate, True, predicate)
    return annotate_with_constraint(TypedValue(bool), constraint)


def _isinstance_impl(ctx: CallContext) -> Value:
    class_or_tuple = ctx.vars["class_or_tuple"]
    varname = ctx.varname_for_arg("obj")
    if varname is None or not isinstance(class_or_tuple, KnownValue):
        return TypedValue(bool)
    try:
        narrowed_types = list(_resolve_isinstance_arg(class_or_tuple.val))
    except _CannotResolve as e:
        ctx.show_error(
            f'Second argument to "isinstance" must be a type, union,'
            f' or tuple of types, not "{e.args[0]!r}"',
            ErrorCode.incompatible_argument,
            arg="class_or_tuple",
        )
        return TypedValue(bool)
    narrowed_type = unite_values(*[TypedValue(typ) for typ in narrowed_types])
    predicate = IsAssignablePredicate(narrowed_type, ctx.visitor, positive_only=False)
    constraint = Constraint(varname, ConstraintType.predicate, True, predicate)
    return annotate_with_constraint(TypedValue(bool), constraint)


class _CannotResolve(Exception):
    pass


def _resolve_isinstance_arg(val: object) -> Iterable[type]:
    if safe_isinstance(val, type):
        yield val
    elif safe_isinstance(val, tuple):
        for elt in val:
            yield from _resolve_isinstance_arg(elt)
    else:
        origin = typing_extensions.get_origin(val)
        if is_union(origin):
            for arg in typing_extensions.get_args(val):
                yield from _resolve_isinstance_arg(arg)
        elif safe_isinstance(origin, type):
            yield origin
        else:
            raise _CannotResolve(val)


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


def _sequence_common_getitem_impl(ctx: CallContext, typ: type) -> ImplReturn:
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
                        elif typ is tuple:
                            ctx.show_error(f"Tuple index out of range: {key}")
                            return AnyValue(AnySource.error)
                        else:
                            # fall back to the common type
                            return self_value.args[0]
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
                elif self_value.typ in (list, tuple, collections.abc.Sequence):
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
    return _sequence_common_getitem_impl(ctx, list)


def _tuple_getitem_impl(ctx: CallContext) -> ImplReturn:
    return _sequence_common_getitem_impl(ctx, tuple)


# This one seems to be needed because Sequence.__getitem__ gets confused with the __getitem__
# for various internal typing classes.
def _sequence_getitem_impl(ctx: CallContext) -> ImplReturn:
    return _sequence_common_getitem_impl(ctx, collections.abc.Sequence)


def _typeddict_setitem(
    self_value: TypedDictValue, key: Value, value: Value, ctx: CallContext
) -> None:
    if not isinstance(key, KnownValue):
        if not TypedValue(str).is_assignable(key, ctx.visitor):
            ctx.show_error(
                f"TypedDict key must be str, not {key}",
                ErrorCode.invalid_typeddict_key,
                arg="k",
            )
            return
        if self_value.extra_keys is None or self_value.extra_keys is NO_RETURN_VALUE:
            ctx.show_error(
                f"Cannot set unknown key {key} in TypedDict {self_value}",
                ErrorCode.invalid_typeddict_key,
                arg="k",
            )
            return
        for td_key, entry in self_value.items.items():
            if not key.is_assignable(KnownValue(td_key), ctx.visitor):
                continue
            if entry.readonly:
                ctx.show_error(
                    f"Cannot set readonly key {key} in TypedDict {self_value}",
                    ErrorCode.readonly_typeddict,
                    arg="k",
                )
                return
            can_assign = entry.typ.can_assign(value, ctx.visitor)
            if isinstance(can_assign, CanAssignError):
                ctx.show_error(
                    f"Value for key {key} must be {entry.typ}, not {value}",
                    ErrorCode.incompatible_argument,
                    arg="v",
                    detail=str(can_assign),
                )
                return
        return

    if not isinstance(key.val, str):
        ctx.show_error(
            f"TypedDict key must be a string literal (got {key})",
            ErrorCode.invalid_typeddict_key,
            arg="k",
        )
        return
    if key.val not in self_value.items:
        if self_value.extra_keys_readonly:
            ctx.show_error(
                f"Cannot set unknown key {key.val!r} in closed TypedDict {self_value}",
                ErrorCode.readonly_typeddict,
                arg="k",
            )
            return
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
        entry = self_value.items[key.val]
        if entry.readonly:
            ctx.show_error(
                f"Cannot set readonly key {key.val!r} in TypedDict {self_value}",
                ErrorCode.readonly_typeddict,
                arg="k",
            )
            return
        expected_type = entry.typ
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
                    entry = self_value.items[key.val]
                    return entry.typ
                # probably KeyError, but catch anything in case it's an
                # unhashable str subclass or something
                except Exception:
                    pass
            if (
                self_value.extra_keys is not None
                and self_value.extra_keys is not NO_RETURN_VALUE
            ):
                return self_value.extra_keys
            if isinstance(key, KnownValue):
                ctx.show_error(
                    f"Unknown TypedDict key {key.val!r}",
                    ErrorCode.invalid_typeddict_key,
                    arg="k",
                )
            else:
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
                    entry = self_value.items[key.val]
                # probably KeyError, but catch anything in case it's an
                # unhashable str subclass or something
                except Exception:
                    pass
                else:
                    if entry.required:
                        return entry.typ
                    else:
                        return entry.typ | default
            if (
                self_value.extra_keys is not None
                and self_value.extra_keys is not NO_RETURN_VALUE
            ):
                return self_value.extra_keys | default
            if isinstance(key, KnownValue):
                ctx.show_error(
                    f"Unknown TypedDict key {key.val!r}",
                    ErrorCode.invalid_typeddict_key,
                    arg="k",
                )
            else:
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


def _dict_delitem_impl(ctx: CallContext) -> ImplReturn:
    key = ctx.vars["key"]
    varname = ctx.varname_for_arg("self")
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
                entry = self_value.items[key.val]
            # probably KeyError, but catch anything in case it's an
            # unhashable str subclass or something
            except Exception:
                pass
            else:
                if entry.readonly:
                    ctx.show_error(
                        f"Cannot delete readonly TypedDict key {key}",
                        error_code=ErrorCode.readonly_typeddict,
                        arg="key",
                    )
                elif entry.required:
                    ctx.show_error(
                        f"Cannot delete required TypedDict key {key}",
                        error_code=ErrorCode.incompatible_argument,
                        arg="key",
                    )
                return ImplReturn(KnownValue(None))
        if self_value.extra_keys_readonly:
            ctx.show_error(
                f"Cannot delete unknown key {key} in closed TypedDict {self_value}",
                ErrorCode.readonly_typeddict,
                arg="key",
            )
        elif self_value.extra_keys is None or self_value.extra_keys is NO_RETURN_VALUE:
            ctx.show_error(
                f"Key {key} does not exist in TypedDict",
                ErrorCode.invalid_typeddict_key,
                arg="key",
            )
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
            # No error; it might have been added where we couldn't see it
            return ImplReturn(KnownValue(None))
        return ImplReturn(KnownValue(None), no_return_unless=no_return_unless)
    elif isinstance(self_value, TypedValue):
        key_type = self_value.get_generic_arg_for_type(dict, ctx.visitor, 0)
        tv_map = key_type.can_assign(key, ctx.visitor)
        if isinstance(tv_map, CanAssignError):
            ctx.show_error(
                f"Key {key} is not valid for {self_value}",
                ErrorCode.incompatible_argument,
                arg="key",
            )
    return ImplReturn(KnownValue(None))


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
                entry = self_value.items[key.val]
            # probably KeyError, but catch anything in case it's an
            # unhashable str subclass or something
            except Exception:
                pass
            else:
                if entry.required:
                    ctx.show_error(
                        f"Cannot pop required TypedDict key {key}",
                        error_code=ErrorCode.incompatible_argument,
                        arg="key",
                    )
                elif entry.readonly:
                    ctx.show_error(
                        f"Cannot pop readonly TypedDict key {key}",
                        error_code=ErrorCode.readonly_typeddict,
                        arg="key",
                    )
                return ImplReturn(_maybe_unite(entry.typ, default))
        if self_value.extra_keys_readonly:
            ctx.show_error(
                f"Cannot pop unknown key {key} in closed TypedDict {self_value}",
                ErrorCode.readonly_typeddict,
                arg="key",
            )
            return ImplReturn(
                _maybe_unite(self_value.extra_keys or TypedValue(object), default)
            )
        if (
            self_value.extra_keys is not None
            and self_value.extra_keys is not NO_RETURN_VALUE
        ):
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
                entry = self_value.items[key.val]
            # probably KeyError, but catch anything in case it's an
            # unhashable str subclass or something
            except Exception:
                pass
            else:
                if entry.readonly:
                    ctx.show_error(
                        f"Cannot setdefault readonly TypedDict key {key}",
                        error_code=ErrorCode.readonly_typeddict,
                        arg="key",
                    )
                tv_map = entry.typ.can_assign(default, ctx.visitor)
                if isinstance(tv_map, CanAssignError):
                    ctx.show_error(
                        f"TypedDict key {key.val} expected value of type"
                        f" {entry.typ}, not {default}",
                        ErrorCode.incompatible_argument,
                        arg="default",
                    )
                return ImplReturn(entry.typ)
        if (
            self_value.extra_keys is not None
            and self_value.extra_keys is not NO_RETURN_VALUE
        ):
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
            "Value argument to assert_is_value must be a KnownValue (got"
            f" {expected_value!r}; object is {obj!r})",
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


def _is_assignable_impl(ctx: CallContext) -> Value:
    typ = ctx.vars["typ"]
    if not isinstance(typ, KnownValue):
        return TypedValue(bool)
    guarded_type = type_from_value(typ.val, ctx.visitor)
    varname = ctx.varname_for_arg("value")
    if varname is None:
        return TypedValue(bool)
    return annotate_with_constraint(
        TypedValue(bool),
        Constraint(varname, ConstraintType.is_value_object, True, guarded_type),
    )


def _reveal_type_impl(ctx: CallContext) -> Value:
    value = ctx.vars["value"]
    if ctx.visitor._is_checking():
        message = f"Revealed type is {ctx.visitor.display_value(value)}"
        ctx.show_error(message, ErrorCode.reveal_type, arg="value")
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
            ErrorCode.reveal_type,
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
        ctx.show_error(message, ErrorCode.reveal_type, arg="value")
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
        for key, entry in kwargs_value.items.items():
            if entry.required:
                kwargs[key] = entry.typ
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


def _recursive_unanotate(val: Value) -> Value:
    # Maybe we should also recurse into type parameters?
    if isinstance(val, AnnotatedValue):
        return _recursive_unanotate(val.value)
    elif isinstance(val, MultiValuedValue):
        return unite_values(*[_recursive_unanotate(subval) for subval in val.vals])
    else:
        return val


def _assert_type_impl(ctx: CallContext) -> Value:
    val = _recursive_unanotate(ctx.vars["val"])
    typ = ctx.vars["typ"]
    expected_type = type_from_value(typ, visitor=ctx.visitor, node=ctx.node)
    # Can't distinguish between different kinds of Any here
    if (
        isinstance(val, AnyValue)
        and isinstance(expected_type, AnyValue)
        and expected_type.source is not AnySource.error
    ):
        return val
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


def len_transformer(val: Value, op: Type[ast.AST], comparator: object) -> Value:
    if not isinstance(comparator, int):
        return val
    if isinstance(len_of_value(val), KnownValue):
        return val  # no need to specify
    if op is ast.Eq:
        return annotate_value(
            val,
            [
                CustomCheckExtension(MinLen(comparator)),
                CustomCheckExtension(MaxLen(comparator)),
            ],
        )
    elif op is ast.Lt:
        return annotate_value(val, [CustomCheckExtension(MaxLen(comparator - 1))])
    elif op is ast.LtE:
        return annotate_value(val, [CustomCheckExtension(MaxLen(comparator))])
    elif op is ast.Gt:
        return annotate_value(val, [CustomCheckExtension(MinLen(comparator + 1))])
    elif op is ast.GtE:
        return annotate_value(val, [CustomCheckExtension(MinLen(comparator))])
    else:
        return val


def _len_impl(ctx: CallContext) -> ImplReturn:
    varname = ctx.varname_for_arg("obj")
    if varname is None:
        constraint = NULL_CONSTRAINT
    else:
        constraint = PredicateProvider(varname, len_of_value, len_transformer)
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


# Any has a __call__ method at runtime that always raises.
def _any_impl(ctx: CallContext) -> Value:
    ctx.show_error("Any is not callable. Maybe you meant cast(Any, ...)?")
    return AnyValue(AnySource.error)


# Should not be necessary, but by default we pick up a wrong signature for
# typing.NamedTuple
def _namedtuple_impl(ctx: CallContext) -> Value:
    has_kwargs = (
        isinstance(ctx.vars["kwargs"], TypedDictValue)
        and len(ctx.vars["kwargs"].items) > 0
    )
    # Mirrors the runtime logic in typing.NamedTuple in 3.13
    if ctx.vars["fields"] is _NO_ARG_SENTINEL:
        if has_kwargs:
            ctx.show_error(
                'Creating "NamedTuple" classes using keyword arguments'
                " is deprecated and will be disallowed in Python 3.15. "
                "Use the class-based or functional syntax instead.",
                ErrorCode.deprecated,
                arg="kwargs",
            )
        else:
            ctx.show_error(
                'Failing to pass a value for the "fields" parameter'
                " is deprecated and will be disallowed in Python 3.15.",
                ErrorCode.deprecated,
            )
    elif ctx.vars["fields"] == KnownValue(None):
        if has_kwargs:
            ctx.show_error(
                'Cannot pass "None" as the "fields" parameter '
                "and also specify fields using keyword arguments",
                ErrorCode.incompatible_call,
                arg="fields",
            )
        else:
            ctx.show_error(
                'Passing "None" as the "fields" parameter '
                " is deprecated and will be disallowed in Python 3.15.",
                ErrorCode.deprecated,
                arg="fields",
            )
    elif has_kwargs:
        ctx.show_error(
            "Either list of fields or keywords"
            ' can be provided to "NamedTuple", not both',
            ErrorCode.incompatible_call,
            arg="kwargs",
        )

    return AnyValue(AnySource.inference)


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
            return_annotation=KnownValue(None),
            impl=_assert_is_value_impl,
            callable=assert_is_value,
        ),
        Signature.make(
            [SigParameter("value"), SigParameter("typ")],
            return_annotation=TypedValue(bool),
            impl=_is_assignable_impl,
            callable=runtime.is_assignable,
        ),
        Signature.make(
            [SigParameter("value"), SigParameter("typ")],
            return_annotation=TypedValue(bool),
            impl=_is_assignable_impl,
            callable=runtime.is_compatible,  # static analysis: ignore[deprecated]
        ),
        Signature.make(
            [SigParameter("value", _POS_ONLY, annotation=TypeVarValue(T))],
            return_annotation=TypeVarValue(T),
            impl=_reveal_type_impl,
            callable=reveal_type,
        ),
        Signature.make(
            [],
            return_annotation=KnownValue(None),
            impl=_reveal_locals_impl,
            callable=reveal_locals,
        ),
        Signature.make(
            [SigParameter("value", _POS_ONLY, annotation=TypeVarValue(T))],
            return_annotation=TypeVarValue(T),
            impl=_dump_value_impl,
            callable=dump_value,
        ),
        # builtins
        Signature.make(
            [SigParameter("self", _POS_ONLY)],
            callable=type.__subclasses__,
            impl=_subclasses_impl,
            return_annotation=GenericValue(list, [TypedValue(type)]),
        ),
        Signature.make(
            [SigParameter("obj", _POS_ONLY), SigParameter("class_or_tuple", _POS_ONLY)],
            impl=_isinstance_impl,
            callable=isinstance,
            return_annotation=TypedValue(bool),
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
            return_annotation=TypedValue(bool),
        ),
        Signature.make(
            [SigParameter("obj"), SigParameter("class_or_tuple")],
            impl=_isinstance_impl,
            callable=safe_isinstance,
            return_annotation=TypedValue(bool),
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
            return_annotation=TypedValue(bool),
        ),
        Signature.make(
            [
                SigParameter("object", _POS_ONLY),
                SigParameter("name", _POS_ONLY, annotation=TypedValue(str)),
                SigParameter("default", _POS_ONLY, default=_NO_ARG_SENTINEL),
            ],
            return_annotation=AnyValue(AnySource.inference),
            callable=getattr,
        ),
        Signature.make(
            [
                SigParameter("object", _POS_ONLY),
                SigParameter("name", _POS_ONLY, annotation=TypedValue(str)),
            ],
            impl=_hasattr_impl,
            callable=hasattr,
            return_annotation=TypedValue(bool),
        ),
        Signature.make(
            [
                SigParameter("object", _POS_ONLY),
                SigParameter("name", _POS_ONLY, annotation=TypedValue(str)),
            ],
            impl=_hasattr_impl,
            callable=hasattr_static,
            return_annotation=TypedValue(bool),
        ),
        Signature.make(
            [
                SigParameter("object", _POS_ONLY),
                SigParameter("name", _POS_ONLY, annotation=TypedValue(str)),
                SigParameter("value", _POS_ONLY),
            ],
            impl=_setattr_impl,
            callable=setattr,
            return_annotation=KnownValue(None),
        ),
        Signature.make(
            [
                SigParameter("type", _POS_ONLY, default=_NO_ARG_SENTINEL),
                SigParameter("obj", _POS_ONLY, default=_NO_ARG_SENTINEL),
            ],
            impl=_super_impl,
            callable=super,
            return_annotation=TypedValue(super),
        ),
        Signature.make(
            [SigParameter("iterable", _POS_ONLY, default=_NO_ARG_SENTINEL)],
            impl=_tuple_impl,
            callable=tuple,
            return_annotation=TypedValue(tuple),
        ),
        Signature.make(
            [SigParameter("iterable", _POS_ONLY, default=_NO_ARG_SENTINEL)],
            impl=_list_impl,
            callable=list,
            return_annotation=TypedValue(list),
        ),
        Signature.make(
            [SigParameter("iterable", _POS_ONLY, default=_NO_ARG_SENTINEL)],
            impl=_set_impl,
            callable=set,
            return_annotation=TypedValue(set),
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(list)),
                SigParameter("object", _POS_ONLY),
            ],
            callable=list.append,
            impl=_list_append_impl,
            return_annotation=KnownValue(None),
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(list)),
                SigParameter("x", _POS_ONLY, annotation=TypedValue(list)),
            ],
            callable=list.__add__,
            impl=_list_add_impl,
            return_annotation=TypedValue(list),
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
            return_annotation=TypedValue(list),
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
            return_annotation=KnownValue(None),
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(list)),
                SigParameter("obj", _POS_ONLY),
            ],
            callable=list.__getitem__,
            impl=_list_getitem_impl,
            return_annotation=AnyValue(AnySource.inference),
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(tuple)),
                SigParameter("obj", _POS_ONLY),
            ],
            callable=tuple.__getitem__,
            impl=_tuple_getitem_impl,
            return_annotation=AnyValue(AnySource.inference),
        ),
        Signature.make(
            [
                SigParameter(
                    "self", _POS_ONLY, annotation=TypedValue(collections.abc.Sequence)
                ),
                SigParameter("obj", _POS_ONLY),
            ],
            callable=collections.abc.Sequence.__getitem__,
            impl=_sequence_getitem_impl,
            return_annotation=AnyValue(AnySource.inference),
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(set)),
                SigParameter("object", _POS_ONLY),
            ],
            callable=set.add,
            impl=_set_add_impl,
            return_annotation=KnownValue(None),
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(dict)),
                SigParameter("k", _POS_ONLY),
                SigParameter("v", _POS_ONLY),
            ],
            callable=dict.__setitem__,
            impl=_dict_setitem_impl,
            return_annotation=KnownValue(None),
        ),
        Signature.make(
            [
                SigParameter(
                    "self",
                    _POS_ONLY,
                    annotation=GenericValue(dict, [TypeVarValue(K), TypeVarValue(V)]),
                ),
                SigParameter("k", _POS_ONLY),
            ],
            callable=dict.__getitem__,
            impl=_dict_getitem_impl,
            return_annotation=TypeVarValue(V),
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(dict)),
                SigParameter("key", _POS_ONLY),
                SigParameter("default", _POS_ONLY, default=KnownValue(None)),
            ],
            callable=dict.get,
            impl=_dict_get_impl,
            return_annotation=AnyValue(AnySource.inference),
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(dict)),
                SigParameter("key", _POS_ONLY),
                SigParameter("default", _POS_ONLY, default=KnownValue(None)),
            ],
            callable=dict.setdefault,
            impl=_dict_setdefault_impl,
            return_annotation=AnyValue(AnySource.inference),
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(dict)),
                SigParameter("key", _POS_ONLY),
                SigParameter("default", _POS_ONLY, default=_NO_ARG_SENTINEL),
            ],
            callable=dict.pop,
            impl=_dict_pop_impl,
            return_annotation=AnyValue(AnySource.inference),
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(dict)),
                SigParameter("key", _POS_ONLY),
            ],
            callable=dict.__delitem__,
            impl=_dict_delitem_impl,
            return_annotation=KnownValue(None),
        ),
        Signature.make(
            [
                SigParameter("self", _POS_ONLY, annotation=TypedValue(dict)),
                SigParameter("m", _POS_ONLY, default=_NO_ARG_SENTINEL),
                SigParameter("kwargs", ParameterKind.VAR_KEYWORD),
            ],
            return_annotation=KnownValue(None),
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
            return_annotation=DictIncompleteValue(
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
            return_annotation=TypedValue(str),
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
            return_annotation=TypedValue(bytes),
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
            return_annotation=TypedValue(str),
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
            return_annotation=TypeVarValue(T),
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
            return_annotation=KnownValue(None),
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
            return_annotation=KnownValue(None),
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
            return_annotation=KnownValue(None),
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
            return_annotation=TypedValue(int),
        ),
        Signature.make(
            [
                SigParameter(
                    "o", ParameterKind.POSITIONAL_ONLY, default=_NO_ARG_SENTINEL
                )
            ],
            callable=bool,
            impl=_bool_impl,
            return_annotation=TypedValue(bool),
        ),
        Signature.make(
            [
                SigParameter("args", ParameterKind.VAR_POSITIONAL),
                SigParameter("kwargs", ParameterKind.VAR_KEYWORD),
            ],
            callable=typing.Any,
            impl=_any_impl,
            return_annotation=AnyValue(AnySource.error),
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
            sig = Signature.make(
                [SigParameter("value", _POS_ONLY, annotation=TypeVarValue(T))],
                return_annotation=TypeVarValue(T),
                impl=_reveal_type_impl,
                callable=reveal_type_func,
            )
            signatures.append(sig)
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
                return_annotation=TypeVarValue(T),
                callable=assert_type_func,
                impl=_assert_type_impl,
            )
            signatures.append(sig)
        try:
            namedtuple_func = getattr(mod, "NamedTuple")
        except AttributeError:
            pass
        else:
            sig = Signature.make(
                [
                    SigParameter("typename", _POS_ONLY, annotation=TypedValue(str)),
                    SigParameter(
                        "fields",
                        _POS_ONLY,
                        annotation=GenericValue(
                            collections.abc.Iterable,
                            [
                                SequenceValue(
                                    tuple,
                                    [
                                        (False, TypedValue(str)),
                                        (False, AnyValue(AnySource.inference)),
                                    ],
                                )
                            ],
                        )
                        | KnownValue(None),
                        default=_NO_ARG_SENTINEL,
                    ),
                    SigParameter("kwargs", ParameterKind.VAR_KEYWORD),
                ],
                return_annotation=TypedValue(type),
                callable=namedtuple_func,
                impl=_namedtuple_impl,
                allow_call=True,
            )
            signatures.append(sig)
    return {sig.callable: sig for sig in signatures}
