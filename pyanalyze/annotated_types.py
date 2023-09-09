"""

Support for annotations from the annotated_types library.

"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional, Union
from typing_extensions import Literal

from annotated_types import BaseMetadata

from pyanalyze.value import CanAssign, CanAssignContext, Value, flatten_values

from .extensions import CustomCheck
from .value import (
    AnnotatedValue,
    AnyValue,
    CanAssignError,
    CustomCheckExtension,
    DictIncompleteValue,
    KnownValue,
    SequenceValue,
    TypedDictValue,
    unannotate,
)

try:
    import annotated_types
except ImportError:

    def get_annotated_types_extension(obj: object) -> Iterable[CustomCheckExtension]:
        return []

else:

    def get_annotated_types_extension(obj: object) -> Iterable[CustomCheckExtension]:
        if not isinstance(obj, annotated_types.BaseMetadata):
            return
        if isinstance(obj, annotated_types.GroupedMetadata):
            for value in obj:
                maybe_ext = _get_single_annotated_types_extension(value)
                if maybe_ext is not None:
                    yield CustomCheckExtension(maybe_ext)
        else:
            maybe_ext = _get_single_annotated_types_extension(obj)
            if maybe_ext is not None:
                yield CustomCheckExtension(maybe_ext)

    def _get_single_annotated_types_extension(
        obj: annotated_types.BaseMetadata,
    ) -> Optional[CustomCheck]:
        if isinstance(obj, annotated_types.Gt):
            return GtCheck(obj, obj.gt)
        elif isinstance(obj, annotated_types.Ge):
            return GeCheck(obj, obj.ge)
        elif isinstance(obj, annotated_types.Lt):
            return LtCheck(obj, obj.lt)
        elif isinstance(obj, annotated_types.Le):
            return LeCheck(obj, obj.le)
        elif isinstance(obj, annotated_types.MultipleOf):
            return MultipleOfCheck(obj, obj.multiple_of)
        elif isinstance(obj, annotated_types.MinLen):
            return MinLenCheck(obj, obj.min_length)
        elif isinstance(obj, annotated_types.MaxLen):
            return MaxLenCheck(obj, obj.max_length)
        elif isinstance(obj, annotated_types.Timezone):
            return TimezoneCheck(obj, obj.tz)
        elif isinstance(obj, annotated_types.Predicate):
            return PredicateCheck(obj, obj.func)
        else:
            # In case annotated_types adds more kinds of checks in the future
            return None

    @dataclass
    class AnnotatedTypesCheck(CustomCheck):
        metadata: annotated_types.BaseMetadata

        def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
            for subval in flatten_values(value):
                original_subval = subval
                if isinstance(subval, AnnotatedValue):
                    if any(
                        ext.metadata == self.metadata
                        or self.is_compatible_metadata(ext.metadata)
                        for ext in subval.get_custom_check_of_type(AnnotatedTypesCheck)
                    ):
                        continue
                subval = unannotate(subval)
                if isinstance(subval, AnyValue):
                    continue
                if isinstance(subval, KnownValue):
                    try:
                        result = self.predicate(subval.val)
                    except Exception:
                        return CanAssignError(
                            f"Value {subval.val} cannot be checked against predicate"
                            f" {self.metadata}"
                        )
                    else:
                        if not result:
                            return CanAssignError(
                                f"Value {subval.val} does not match predicate"
                                f" {self.metadata}"
                            )
                else:
                    can_assign = self.can_assign_non_literal(original_subval)
                    if isinstance(can_assign, CanAssignError):
                        return can_assign
            return {}

        def predicate(self, value: Any) -> bool:
            raise NotImplementedError

        def is_compatible_metadata(self, metadata: BaseMetadata) -> bool:
            """Override this to allow metadata that is not exactly the same as the
            one in this object to match. For example, Gt(5) should accept Gt(4), as that
            is a strictly weaker condition.
            """
            return False

        def can_assign_non_literal(self, value: Value) -> CanAssign:
            """Override this to allow some non-Literal values. For example, the
            MinLen extensions can allow sequences with sufficiently known lengths."""
            return CanAssignError(
                f"Cannot determine whether {value} fulfills predicate {self.metadata}"
            )

    @dataclass
    class GtCheck(AnnotatedTypesCheck):
        value: Any

        def predicate(self, value: Any) -> bool:
            return value > self.value

        def is_compatible_metadata(self, metadata: BaseMetadata) -> bool:
            if isinstance(metadata, annotated_types.Gt):
                return metadata.gt >= self.value
            elif isinstance(metadata, annotated_types.Ge):
                return metadata.ge > self.value
            else:
                return False

    @dataclass
    class GeCheck(AnnotatedTypesCheck):
        value: Any

        def predicate(self, value: Any) -> bool:
            return value >= self.value

        def is_compatible_metadata(self, metadata: BaseMetadata) -> bool:
            if isinstance(metadata, annotated_types.Gt):
                return metadata.gt > self.value
            elif isinstance(metadata, annotated_types.Ge):
                return metadata.ge >= self.value
            else:
                return False

    @dataclass
    class LtCheck(AnnotatedTypesCheck):
        value: Any

        def predicate(self, value: Any) -> bool:
            return value < self.value

        def is_compatible_metadata(self, metadata: BaseMetadata) -> bool:
            if isinstance(metadata, annotated_types.Lt):
                return metadata.lt <= self.value
            elif isinstance(metadata, annotated_types.Le):
                return metadata.le < self.value
            else:
                return False

    @dataclass
    class LeCheck(AnnotatedTypesCheck):
        value: Any

        def predicate(self, value: Any) -> bool:
            return value <= self.value

        def is_compatible_metadata(self, metadata: BaseMetadata) -> bool:
            if isinstance(metadata, annotated_types.Lt):
                return metadata.lt < self.value
            elif isinstance(metadata, annotated_types.Le):
                return metadata.le <= self.value
            else:
                return False

    @dataclass
    class MultipleOfCheck(AnnotatedTypesCheck):
        value: Any

        def predicate(self, value: Any) -> bool:
            return value % self.value == 0

        def is_compatible_metadata(self, metadata: BaseMetadata) -> bool:
            if isinstance(metadata, annotated_types.MultipleOf):
                # If we want a MultipleOf(10), but we're passed a MultipleOf(5), that's ok
                return self.value % metadata.multiple_of == 0
            else:
                return False

    @dataclass
    class MinLenCheck(AnnotatedTypesCheck):
        value: Any

        def predicate(self, value: Any) -> bool:
            return len(value) >= self.value

        def is_compatible_metadata(self, metadata: BaseMetadata) -> bool:
            if isinstance(metadata, annotated_types.MinLen):
                return metadata.min_length <= self.value
            else:
                return False

        def can_assign_non_literal(self, value: Value) -> CanAssign:
            min_len = _min_len_of_value(value)
            if min_len is not None and min_len >= self.value:
                return {}
            return super().can_assign_non_literal(value)

    @dataclass
    class MaxLenCheck(AnnotatedTypesCheck):
        value: Any

        def predicate(self, value: Any) -> bool:
            return len(value) <= self.value

        def is_compatible_metadata(self, metadata: BaseMetadata) -> bool:
            if isinstance(metadata, annotated_types.MaxLen):
                return metadata.max_length >= self.value
            else:
                return False

        def can_assign_non_literal(self, value: Value) -> CanAssign:
            max_len = _max_len_of_value(value)
            if max_len is not None and max_len <= self.value:
                return {}
            return super().can_assign_non_literal(value)

    @dataclass
    class TimezoneCheck(AnnotatedTypesCheck):
        value: Union[str, timezone, Literal[None, ...]]

        def predicate(self, value: Any) -> bool:
            if not isinstance(value, datetime):
                return False
            if self.value is None:
                return value.tzinfo is None
            elif self.value is ...:
                return value.tzinfo is not None
            elif isinstance(self.value, timezone):
                return value.tzinfo == self.value
            else:
                return False

        def is_compatible_metadata(self, metadata: BaseMetadata) -> bool:
            if (
                self.value is ...
                and isinstance(metadata, annotated_types.Timezone)
                and metadata.value is not None
            ):
                return True
            return False

    @dataclass
    class PredicateCheck(AnnotatedTypesCheck):
        predicate_callable: Callable[[Any], bool]

        def predicate(self, value: Any) -> bool:
            return self.predicate_callable(value)


def _min_len_of_value(val: Value) -> Optional[int]:
    if isinstance(val, SequenceValue):
        return sum(is_many is False for is_many, _ in val.members)
    elif isinstance(val, DictIncompleteValue):
        return sum(pair.is_required and not pair.is_many for pair in val.kv_pairs)
    elif isinstance(val, TypedDictValue):
        return sum(required for required, _ in val.items.values())
    else:
        return None


def _max_len_of_value(val: Value) -> Optional[int]:
    if isinstance(val, SequenceValue):
        maximum = 0
        for is_many, _ in val.members:
            if is_many:
                return None
            maximum += 1
        return maximum
    elif isinstance(val, DictIncompleteValue):
        maximum = 0
        for pair in val.kv_pairs:
            if pair.is_many:
                return None
            if pair.is_required:
                maximum += 1
        return maximum
    else:
        # Always None for TypedDicts as TypedDicts may have arbitrary extra keys
        return None
