"""

Reusable predicates.

"""
from dataclasses import dataclass
import enum
import operator
from typing import Optional

from .safe import safe_issubclass
from .value import (
    KnownValue,
    TypedValue,
    Value,
    unite_values,
    CanAssignContext,
    is_overlapping,
    unannotate,
)


@dataclass
class IsAssignablePredicate:
    """Predicate that filters out values that are not assignable to pattern_value."""

    pattern_value: Value
    ctx: CanAssignContext
    positive_only: bool

    def __call__(self, value: Value, positive: bool) -> Optional[Value]:
        compatible = is_overlapping(self.pattern_value, value, self.ctx)
        if positive:
            if not compatible:
                return None
            if value.is_assignable(self.pattern_value, self.ctx):
                return self.pattern_value
        elif not self.positive_only:
            if compatible:
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
