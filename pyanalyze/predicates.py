"""

Reusable predicates.

"""

import enum
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from .safe import safe_issubclass
from .value import (
    NO_RETURN_VALUE,
    AnnotatedValue,
    AnyValue,
    CanAssignContext,
    KnownValue,
    MultiValuedValue,
    SubclassValue,
    TypedValue,
    TypeVarValue,
    Value,
    is_overlapping,
    unannotate,
    unite_values,
)


def is_universally_assignable(value: Value, target_value: Value) -> bool:
    if value is NO_RETURN_VALUE or isinstance(value, AnyValue):
        return True
    elif value == TypedValue(type) and isinstance(target_value, SubclassValue):
        return True
    elif isinstance(value, AnnotatedValue):
        return is_universally_assignable(value.value, target_value)
    elif isinstance(value, MultiValuedValue):
        return all(
            is_universally_assignable(subval, target_value) for subval in value.vals
        )
    elif isinstance(value, TypeVarValue):
        return True
    return False


@dataclass
class IsAssignablePredicate:
    """Predicate that filters out values that are not assignable to pattern_value.

    This only works reliably for simple pattern_values, such as TypedValue.

    """

    pattern_value: Value
    ctx: CanAssignContext
    positive_only: bool

    def __call__(self, value: Value, positive: bool) -> Optional[Value]:
        compatible = is_overlapping(self.pattern_value, value, self.ctx)
        if positive:
            if not compatible:
                return None
            if self.pattern_value.is_assignable(value, self.ctx):
                if is_universally_assignable(value, unannotate(self.pattern_value)):
                    return self.pattern_value
                return value
            else:
                return self.pattern_value
        elif not self.positive_only:
            if self.pattern_value.is_assignable(
                value, self.ctx
            ) and not is_universally_assignable(value, unannotate(self.pattern_value)):
                return None
        return value


_OPERATOR = {
    (True, True): operator.is_,
    (False, True): operator.is_not,
    (True, False): operator.eq,
    (False, False): operator.ne,
}


@dataclass
class EqualsPredicate:
    """Predicate that filters out values that are not equal to pattern_val."""

    pattern_val: object
    ctx: CanAssignContext
    use_is: bool = False

    def __call__(self, value: Value, positive: bool) -> Optional[Value]:
        inner_value = unannotate(value)
        if isinstance(inner_value, KnownValue):
            op = _OPERATOR[(positive, self.use_is)]
            try:
                result = op(inner_value.val, self.pattern_val)
            except Exception:
                pass
            else:
                if not result:
                    return None
        elif positive:
            known_self = KnownValue(self.pattern_val)
            if value.is_assignable(known_self, self.ctx):
                return known_self
            else:
                return None
        else:
            pattern_type = type(self.pattern_val)
            if pattern_type is bool:
                simplified = unannotate(value)
                if isinstance(simplified, TypedValue) and simplified.typ is bool:
                    return KnownValue(not self.pattern_val)
            elif safe_issubclass(pattern_type, enum.Enum):
                simplified = unannotate(value)
                if isinstance(simplified, TypedValue) and simplified.typ is type(
                    self.pattern_val
                ):
                    return unite_values(
                        *[
                            KnownValue(val)
                            for val in pattern_type
                            if val is not self.pattern_val
                        ]
                    )
        return value


@dataclass
class InPredicate:
    """Predicate that filters out values that are not in pattern_vals."""

    pattern_vals: Sequence[object]
    pattern_type: type
    ctx: CanAssignContext

    def __call__(self, value: Value, positive: bool) -> Optional[Value]:
        inner_value = unannotate(value)
        if isinstance(inner_value, KnownValue):
            try:
                if positive:
                    result = inner_value.val in self.pattern_vals
                else:
                    result = inner_value.val not in self.pattern_vals
            except Exception:
                pass
            else:
                if not result:
                    return None
        elif positive:
            acceptable_values = [
                KnownValue(pattern_val)
                for pattern_val in self.pattern_vals
                if value.is_assignable(KnownValue(pattern_val), self.ctx)
            ]
            if acceptable_values:
                return unite_values(*acceptable_values)
            else:
                return None
        else:
            if safe_issubclass(self.pattern_type, enum.Enum):
                simplified = unannotate(value)
                if (
                    isinstance(simplified, TypedValue)
                    and simplified.typ is self.pattern_type
                ):
                    return unite_values(
                        *[
                            KnownValue(val)
                            for val in self.pattern_type
                            if val not in self.pattern_vals
                        ]
                    )
        return value
