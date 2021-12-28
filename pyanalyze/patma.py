"""

Visitor for pattern matching.

"""

import ast
import collections.abc
from dataclasses import dataclass
import itertools

import qcore
import pyanalyze
from typing import Any, Callable, Optional, Sequence, TypeVar

from pyanalyze.implementation import len_of_value

from .extensions import CustomCheck
from .error_code import ErrorCode
from .stacked_scopes import (
    NULL_CONSTRAINT,
    AbstractConstraint,
    AndConstraint,
    Composite,
    Constraint,
    ConstraintType,
    OrConstraint,
    annotate_with_constraint,
    constrain_value,
)
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CanAssign,
    CanAssignContext,
    CanAssignError,
    CustomCheckExtension,
    SequenceIncompleteValue,
    TypedValue,
    Value,
    KnownValue,
    flatten_values,
    unannotate,
    unpack_values,
)

try:
    from ast import (
        match_case,
        Match,
        MatchAs,
        MatchClass,
        MatchMapping,
        MatchOr,
        MatchSequence,
        MatchSingleton,
        MatchStar,
        MatchValue,
    )
except ImportError:
    # 3.9 and lower
    match_case = Match = MatchAs = MatchClass = MatchMapping = Any
    MatchOr = MatchSequence = MatchSingleton = MatchStar = MatchValue = Any


@dataclass(frozen=True)
class Exclude(CustomCheck):
    """A CustomCheck that excludes certain types."""

    excluded: Value

    def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        for subval in flatten_values(value, unwrap_annotated=True):
            if isinstance(subval, AnyValue):
                continue
            can_assign = self.excluded.can_assign(subval, ctx)
            if not isinstance(can_assign, CanAssignError):
                return CanAssignError(
                    f"{subval} is compatible with excluded type {self.excluded}"
                )
        return {}

    def __str__(self) -> str:
        return f"Exclude[{self.excluded}]"


T = TypeVar("T")
MatchableSequence = AnnotatedValue(
    TypedValue(collections.abc.Sequence),
    [
        CustomCheckExtension(
            Exclude(TypedValue(str) | TypedValue(bytes) | TypedValue(bytearray))
        )
    ],
)


@dataclass
class IsAssignablePredicate:
    pattern_value: Value
    ctx: CanAssignContext

    def __call__(self, value: Value, positive: bool) -> Optional[Value]:
        compatible = is_overlapping(self.pattern_value, value, self.ctx)
        if positive:
            if not compatible:
                return None
            cleaned = unannotate(value)
            if cleaned == TypedValue(object) or isinstance(cleaned, AnyValue):
                return self.pattern_value
        else:
            if compatible:
                return None
        return value


@dataclass
class LenPredicate:
    expected_length: int
    has_star: bool
    ctx: CanAssignContext

    def __call__(self, value: Value, positive: bool) -> Optional[Value]:
        value_len = len_of_value(value)
        if isinstance(value_len, KnownValue) and isinstance(value_len.val, int):
            if self.has_star:
                match = value_len.val >= self.expected_length
            else:
                match = value_len.val == self.expected_length
            if not positive:
                match = not match
            if match:
                return value
            else:
                return None

        cleaned = unannotate(value)
        if (
            not self.has_star
            and isinstance(cleaned, TypedValue)
            and cleaned.typ is tuple
        ):
            # Narrow Tuple[...] to a known length
            arg = cleaned.get_generic_arg_for_type(tuple, self.ctx, 0)
            return SequenceIncompleteValue(
                tuple, [arg for _ in range(self.expected_length)]
            )
        return value


@dataclass
class PatmaVisitor(ast.NodeVisitor):
    visitor: "pyanalyze.name_check_visitor.NameCheckVisitor"

    def visit_MatchSingleton(self, node: MatchSingleton) -> AbstractConstraint:
        self.check_impossible_pattern(node, KnownValue(node.value))
        return self.make_constraint(ConstraintType.is_value, node.value)

    def visit_MatchValue(self, node: MatchValue) -> AbstractConstraint:
        pattern_val = self.visitor.visit(node.value)
        self.check_impossible_pattern(node, pattern_val)
        if not isinstance(pattern_val, KnownValue):
            self.visitor.show_error(
                node,
                f"Match value is not a literal: {pattern_val}",
                ErrorCode.internal_error,
            )
            return NULL_CONSTRAINT

        def predicate_func(value: Value, positive: bool) -> Optional[Value]:
            if isinstance(value, KnownValue):
                try:
                    if positive:
                        result = value.val == pattern_val.val
                    else:
                        result = value.val != pattern_val.val
                except Exception:
                    pass
                else:
                    if not result:
                        return None
            elif positive:
                if value.is_assignable(pattern_val, self.visitor):
                    return pattern_val
                else:
                    return None
            return value

        return self.make_constraint(ConstraintType.predicate, predicate_func)

    def visit_MatchSequence(self, node: MatchSequence) -> AbstractConstraint:
        self.check_impossible_pattern(node, MatchableSequence)
        constraints = [
            self.make_constraint(
                ConstraintType.predicate,
                IsAssignablePredicate(MatchableSequence, self.visitor),
            )
        ]
        starred_index = index_of(node.patterns, lambda pat: isinstance(pat, MatchStar))
        if starred_index is None:
            target_length = len(node.patterns)
            post_starred_length = None
        else:
            target_length = starred_index
            post_starred_length = len(node.patterns) - 1 - target_length
        unpacked = unpack_values(
            self.visitor.match_subject.value,
            self.visitor,
            target_length,
            post_starred_length,
        )
        constraints.append(
            self.make_constraint(
                ConstraintType.predicate,
                LenPredicate(
                    len(node.patterns) - int(starred_index is not None),
                    starred_index is not None,
                    self.visitor,
                ),
            )
        )
        if isinstance(unpacked, CanAssignError):
            unpacked = itertools.repeat(AnyValue(AnySource.generic_argument))
        for pat, subject in zip(node.patterns, unpacked):
            with qcore.override(self.visitor, "match_subject", Composite(subject)):
                constraints.append(self.visit(pat))
        return AndConstraint.make(constraints)

    def visit_MatchStar(self, node: MatchStar) -> AbstractConstraint:
        if node.name is not None:
            self.visitor._set_name_in_scope(
                node.name, node, self.visitor.match_subject.value
            )
        return NULL_CONSTRAINT

    def visit_MatchAs(self, node: MatchAs) -> AbstractConstraint:
        val = self.visitor.match_subject.value
        if node.pattern is None:
            constraint = NULL_CONSTRAINT
        else:
            constraint = self.visit(node.pattern)

        if node.name is not None:
            val = constrain_value(val, constraint)
            self.visitor._set_name_in_scope(node.name, node, val)

        return constraint

    def visit_MatchOr(self, node: MatchOr) -> AbstractConstraint:
        subscopes = []
        constraints = []
        for pattern in node.patterns:
            with self.visitor.scopes.subscope() as subscope:
                constraints.append(self.visit(pattern))
                subscopes.append(subscope)
        self.visitor.scopes.combine_subscopes(subscopes)
        return OrConstraint.make(constraints)

    def generic_visit(self, node: ast.AST) -> AbstractConstraint:
        return NULL_CONSTRAINT

    def make_constraint(self, typ: ConstraintType, value: object) -> AbstractConstraint:
        varname = self.visitor.match_subject.varname
        if varname is None:
            return NULL_CONSTRAINT
        return Constraint(varname, typ, True, value)

    def check_impossible_pattern(self, node: ast.AST, value: Value) -> None:
        if not is_overlapping(self.visitor.match_subject.value, value, self.visitor):
            self.visitor.show_error(
                node,
                f"Impossible pattern: {self.visitor.match_subject.value} can never be"
                f" {value}",
                ErrorCode.impossible_pattern,
            )


def is_overlapping(left: Value, right: Value, ctx: CanAssignContext) -> bool:
    return left.is_assignable(right, ctx) or right.is_assignable(left, ctx)


def index_of(elts: Sequence[T], pred: Callable[[T], bool]) -> Optional[int]:
    for i, elt in enumerate(elts):
        if pred(elt):
            return i
    return None
