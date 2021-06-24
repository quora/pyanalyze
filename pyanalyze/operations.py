"""

Helpers for dealing with common Python operations,
such as iteration.

"""
from itertools import chain
from typing import List, Optional, Sequence, Union, TypeVar

from .value import (
    CallableValue,
    TypeVarValue,
    SequenceIncompleteValue,
    ProtocolValue,
    MultiValuedValue,
    AnnotatedValue,
    UNRESOLVED_VALUE,
    Value,
    GenericValue,
    CanAssignContext,
    CanAssignError,
    replace_known_sequence_value,
    unite_values,
)
from .signature import Signature

T = TypeVar("T")
TValue = TypeVarValue(T)
TV_MAP = {T: TValue}
IterableProto = ProtocolValue(
    "<internal>",
    "Iterable",
    members={"__iter__": (CallableValue(Signature.make([], TValue)), TV_MAP)},
    tv_map=TV_MAP,
)
IteratorProto = ProtocolValue(
    "<internal>",
    "Iterator",
    members={"__next__": (CallableValue(Signature.make([], TValue)), TV_MAP)},
    tv_map=TV_MAP,
)


def is_iterable_value(
    value: Value, ctx: CanAssignContext
) -> Union[Value, CanAssignError]:
    tv_map = IterableProto.can_assign(value, ctx)
    if isinstance(tv_map, CanAssignError):
        return tv_map

    tv_map = IteratorProto.can_assign(tv_map.get(T, UNRESOLVED_VALUE), ctx)
    if isinstance(tv_map, CanAssignError):
        return tv_map
    return tv_map.get(T, UNRESOLVED_VALUE)


def concrete_values_from_iterable(
    value: Value, ctx: CanAssignContext
) -> Union[None, Value, Sequence[Value]]:
    """Return the exact values that can be extracted from an iterable.

    Three possible return types:

    - ``None`` if the argument is not iterable
    - A sequence of :class:`Value` if we know the exact types in the iterable
    - A single :class:`Value` if we just know that the iterable contains this
      value, but not the precise number of them.

    Examples:

    - ``int`` -> ``None``
    - ``tuple[int, str]`` -> ``(int, str)``
    - ``tuple[int, ...]`` -> ``int``

    """
    if isinstance(value, MultiValuedValue):
        subvals = [concrete_values_from_iterable(val, ctx) for val in value.vals]
        if any(subval is None for subval in subvals):
            return None
        value_subvals = [subval for subval in subvals if isinstance(subval, Value)]
        seq_subvals = [
            subval
            for subval in subvals
            if subval is not None and not isinstance(subval, Value)
        ]
        if not value_subvals and len(set(map(len, seq_subvals))) == 1:
            return [unite_values(*vals) for vals in zip(*seq_subvals)]
        return unite_values(*value_subvals, *chain.from_iterable(seq_subvals))
    elif isinstance(value, AnnotatedValue):
        return concrete_values_from_iterable(value.value, ctx)
    value = replace_known_sequence_value(value)
    if isinstance(value, SequenceIncompleteValue) and value.typ is tuple:
        return value.members
    value_or_error = is_iterable_value(value, ctx)
    if isinstance(value_or_error, Value):
        return value_or_error
    return None


def unpack_values(
    value: Value,
    ctx: CanAssignContext,
    target_length: int,
    post_starred_length: Optional[int] = None,
) -> Union[Sequence[Value], CanAssignError]:
    """Implement iterable unpacking.

    If `post_starred_length` is None, return a list of `target_length`
    values, or :class:`CanAssignError` if value is not an iterable of
    the expected length. If `post_starred_length` is not None,
    return a list of `target_length + 1 + post_starred_length` values. This implements
    unpacking like ``a, b, *c, d = ...``.

    """
    if isinstance(value, MultiValuedValue):
        subvals = [
            unpack_values(val, ctx, target_length, post_starred_length)
            for val in value.vals
        ]
        good_subvals = []
        for subval in subvals:
            if isinstance(subval, CanAssignError):
                return CanAssignError(f"Cannot unpack {value}", [subval])
            good_subvals.append(subval)
        if not good_subvals:
            return _create_unpacked_list(
                UNRESOLVED_VALUE, target_length, post_starred_length
            )
        return [unite_values(*vals) for vals in zip(*good_subvals)]
    elif isinstance(value, AnnotatedValue):
        return unpack_values(value.value, ctx, target_length, post_starred_length)
    value = replace_known_sequence_value(value)

    # We treat the different sequence types differently here.
    # - Tuples are  immutable so we can always unpack and show
    #   an error if the length doesn't match.
    # - Sets have randomized order so unpacking into specific values
    #   doesn't make sense. We just fallback to the behavior for
    #   general iterables.
    # - Dicts do have deterministic order but unpacking them doesn't
    #   seem like a common use case. They're also mutable, so if we
    #   did decide to unpack, we'd have to do something similar to
    #   what we do for lists.
    # - Lists can be sensibly unpacked but they are also mutable. Therefore,
    #   we try first to unpack into specific values, and if that doesn't
    #   work due to a length mismatch we fall back to the generic
    #   iterable approach. We experimented both with treating lists
    #   like tuples and with always falling back, and both approaches
    #   led to false positives.
    if isinstance(value, SequenceIncompleteValue):
        if value.typ is tuple:
            return _unpack_value_sequence(
                value, value.members, target_length, post_starred_length
            )
        elif value.typ is list:
            vals = _unpack_value_sequence(
                value, value.members, target_length, post_starred_length
            )
            if not isinstance(vals, CanAssignError):
                return vals

    value_or_error = is_iterable_value(value, ctx)
    if isinstance(value_or_error, CanAssignError):
        return value_or_error
    return _create_unpacked_list(value_or_error, target_length, post_starred_length)


def _create_unpacked_list(
    iterable_type: Value, target_length: int, post_starred_length: Optional[int]
) -> List[Value]:
    if post_starred_length is not None:
        return [
            *([iterable_type] * target_length),
            GenericValue(list, [iterable_type]),
            *([iterable_type] * post_starred_length),
        ]
    else:
        return [iterable_type] * target_length


def _unpack_value_sequence(
    value: Value,
    members: Sequence[Value],
    target_length: int,
    post_starred_length: Optional[int],
) -> Union[Sequence[Value], CanAssignError]:
    actual_length = len(members)
    if post_starred_length is None:
        if actual_length != target_length:
            return CanAssignError(
                f"{value} is of length {actual_length} (expected {target_length})"
            )
        return members
    if actual_length < target_length + post_starred_length:
        return CanAssignError(
            f"{value} is of length {actual_length} (expected at least"
            f" {target_length + post_starred_length})"
        )
    head = members[:target_length]
    if post_starred_length > 0:
        body = SequenceIncompleteValue(
            list, members[target_length:-post_starred_length]
        )
        tail = members[-post_starred_length:]
    else:
        body = SequenceIncompleteValue(list, members[target_length:])
        tail = []
    return [*head, body, *tail]
