"""

Boolability is about whether a value can be used as a boolean.

Objects like functions are always true, so it is likely an error to
use them in a boolean context. This file helps figure out whether
a particular type is usable as a boolean.

"""
import asynq
import enum

from pyanalyze.safe import safe_getattr, safe_hasattr

from .value import (
    KNOWN_MUTABLE_TYPES,
    AnnotatedValue,
    AnyValue,
    DictIncompleteValue,
    KnownValue,
    MultiValuedValue,
    SequenceIncompleteValue,
    SubclassValue,
    TypedDictValue,
    TypedValue,
    UnboundMethodValue,
    Value,
    replace_known_sequence_value,
)


class Boolability(enum.Enum):
    erroring_bool = 1
    """Throws an error if bool() is called on it."""
    boolable = 2
    """Can be safely used as a bool."""
    value_always_false_mutable = 3
    """Always False, but of a mutable type."""
    value_always_true_mutable = 4
    """Always True, but of a mutable type."""
    value_always_false = 5
    """Always False."""
    value_always_true = 6
    """Always True, but of a type that can also be false."""
    type_always_true = 7
    """Value of a type that is always True (because it does not override __bool__)."""

    def is_safely_true(self) -> bool:
        return self in _TRUE_BOOLABILITIES

    def is_safely_false(self) -> bool:
        # We don't treat value_always_false_mutable as safe because
        # empty containers too easily become nonempty.
        return self is Boolability.value_always_false


_TRUE_BOOLABILITIES = {
    Boolability.value_always_true,
    Boolability.value_always_true_mutable,
    Boolability.type_always_true,
}
_FALSE_BOOLABILITIES = {
    Boolability.value_always_false,
    Boolability.value_always_false_mutable,
}
_ASYNQ_BOOL = asynq.FutureBase.__bool__


def get_boolability(value: Value) -> Boolability:
    value = replace_known_sequence_value(value)
    if isinstance(value, MultiValuedValue):
        boolabilities = {_get_boolability_no_mvv(subval) for subval in value.vals}
        if Boolability.erroring_bool in boolabilities:
            return Boolability.erroring_bool
        elif Boolability.boolable in boolabilities:
            return Boolability.boolable
        elif (boolabilities & _TRUE_BOOLABILITIES) and (
            boolabilities & _FALSE_BOOLABILITIES
        ):
            # If it contains both values that are always true and values that are always false,
            # it's boolable.
            return Boolability.boolable
        else:
            # This means the set contains either only truthy or only falsy options.
            # Choose the lowest-valued (and therefore weakest) one.
            return min(boolabilities, key=lambda b: b.value)
    else:
        return _get_boolability_no_mvv(value)


def _get_boolability_no_mvv(value: Value) -> Boolability:
    if isinstance(value, AnnotatedValue):
        value = value.value
    value = replace_known_sequence_value(value)
    if isinstance(value, AnyValue):
        return Boolability.boolable
    elif isinstance(value, UnboundMethodValue):
        if value.secondary_attr_name:
            # Might be anything
            return Boolability.boolable
        else:
            return Boolability.type_always_true
    elif isinstance(value, TypedDictValue):
        if value.num_required_keys():
            # Must be nonempty
            return Boolability.type_always_true
        else:
            return Boolability.boolable
    elif isinstance(value, SequenceIncompleteValue):
        if value.typ is tuple:
            if value.members:
                # We lie slightly here, since at the type level a tuple
                # may be false. But tuples are a common source of boolability
                # bugs and they're rarely mutated, so we put a stronger
                # condition on them.
                return Boolability.type_always_true
            else:
                return Boolability.value_always_false
        else:
            if value.members:
                return Boolability.value_always_true_mutable
            else:
                return Boolability.value_always_false_mutable
    elif isinstance(value, DictIncompleteValue):
        if any(pair.is_required and not pair.is_many for pair in value.kv_pairs):
            return Boolability.value_always_true_mutable
        elif value.kv_pairs:
            return Boolability.boolable
        else:
            return Boolability.value_always_false_mutable
    elif isinstance(value, SubclassValue):
        # Could be false if a metaclass overrides __bool__, but we're
        # not handling that for now.
        return Boolability.type_always_true
    elif isinstance(value, KnownValue):
        try:
            boolean_value = bool(value.val)
        except Exception:
            return Boolability.erroring_bool
        if isinstance(value.val, KNOWN_MUTABLE_TYPES):
            if boolean_value:
                return Boolability.value_always_true_mutable
            else:
                return Boolability.value_always_false_mutable
        type_boolability = _get_type_boolability(type(value.val))
        if boolean_value:
            if type_boolability is Boolability.boolable:
                return Boolability.value_always_true
            elif type_boolability is Boolability.type_always_true:
                return Boolability.type_always_true
            else:
                assert False, (
                    f"inconsistent boolabilities: {boolean_value}, {type_boolability},"
                    f" {value!r}"
                )
        else:
            if type_boolability is Boolability.boolable:
                return Boolability.value_always_false
            else:
                assert False, (
                    f"inconsistent boolabilities: {boolean_value}, {type_boolability},"
                    f" {value!r}"
                )
    elif isinstance(value, TypedValue):
        if isinstance(value.typ, str):
            return Boolability.boolable  # TODO deal with synthetic types
        return _get_type_boolability(value.typ)
    else:
        assert False, f"unhandled value {value!r}"


def _get_type_boolability(typ: type) -> Boolability:
    if safe_hasattr(typ, "__len__"):
        return Boolability.boolable
    dunder_bool = safe_getattr(typ, "__bool__", None)
    if dunder_bool is None:
        return Boolability.type_always_true
    elif dunder_bool is _ASYNQ_BOOL:
        return Boolability.erroring_bool
    else:
        return Boolability.boolable
