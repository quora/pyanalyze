"""

Visitor for pattern matching.

"""

import ast
import collections.abc
from dataclasses import dataclass, replace
import enum
import itertools

import qcore
import pyanalyze
from typing import (
    Any,
    Callable,
    Container,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .implementation import len_of_value
from .signature import MappingValue
from .annotations import type_from_value
from .extensions import CustomCheck
from .error_code import ErrorCode
from .predicates import EqualsPredicate, IsAssignablePredicate
from .stacked_scopes import (
    NULL_CONSTRAINT,
    AbstractConstraint,
    AndConstraint,
    Composite,
    Constraint,
    ConstraintType,
    OrConstraint,
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
    DictIncompleteValue,
    KVPair,
    SequenceIncompleteValue,
    SubclassValue,
    TypedValue,
    Value,
    KnownValue,
    flatten_values,
    kv_pairs_from_mapping,
    replace_known_sequence_value,
    unannotate,
    unite_values,
    unpack_values,
    is_overlapping,
    UNINITIALIZED_VALUE,
)

try:
    from ast import (
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
    MatchAs = MatchClass = MatchMapping = Any
    MatchOr = MatchSequence = MatchSingleton = MatchStar = MatchValue = Any


# For these types, a single class subpattern matches the whole thing
_SPECIAL_CLASS_PATTERN_TYPES = {
    bool,
    bytearray,
    bytes,
    dict,
    float,
    frozenset,
    int,
    list,
    set,
    str,
    tuple,
}
SpecialClassPatternValue = unite_values(
    *[SubclassValue(TypedValue(typ)) for typ in _SPECIAL_CLASS_PATTERN_TYPES]
)


class SpecialPositionalMatch(enum.Enum):
    self = 1  # match against self (special behavior for builtins)
    error = 2  # couldn't figure out the attr, match against Any


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
        return self.make_constraint(
            ConstraintType.predicate,
            EqualsPredicate(node.value, self.visitor, use_is=True),
        )

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

        return self.make_constraint(
            ConstraintType.predicate, EqualsPredicate(pattern_val.val, self.visitor)
        )

    def visit_MatchSequence(self, node: MatchSequence) -> AbstractConstraint:
        self.check_impossible_pattern(node, MatchableSequence)
        starred_index = index_of(node.patterns, lambda pat: isinstance(pat, MatchStar))
        if starred_index is None:
            target_length = len(node.patterns)
            post_starred_length = None
        else:
            target_length = starred_index
            post_starred_length = len(node.patterns) - 1 - target_length
        constraints = [
            self.make_constraint(
                ConstraintType.predicate,
                IsAssignablePredicate(
                    MatchableSequence,
                    self.visitor,
                    positive_only=len(node.patterns) > 1 or starred_index is None,
                ),
            ),
            self.make_constraint(
                ConstraintType.predicate,
                LenPredicate(
                    len(node.patterns) - int(starred_index is not None),
                    starred_index is not None,
                    self.visitor,
                ),
            ),
        ]
        unpacked = unpack_values(
            constrain_value(
                self.visitor.match_subject.value, AndConstraint.make(constraints)
            ),
            self.visitor,
            target_length,
            post_starred_length,
        )
        if isinstance(unpacked, CanAssignError):
            unpacked = itertools.repeat(AnyValue(AnySource.generic_argument))
        for pat, subject in zip(node.patterns, unpacked):
            with qcore.override(self.visitor, "match_subject", Composite(subject)):
                constraints.append(self.visit(pat))
        return AndConstraint.make(constraints)

    def visit_MatchMapping(self, node: MatchMapping) -> AbstractConstraint:
        self.check_impossible_pattern(node, MappingValue)
        constraint = self.make_constraint(
            ConstraintType.predicate,
            IsAssignablePredicate(
                MappingValue, self.visitor, positive_only=len(node.keys) > 0
            ),
        )
        constraints = [constraint]
        subject = constrain_value(self.visitor.match_subject.value, constraint)
        kv_pairs = kv_pairs_from_mapping(subject, self.visitor)
        if isinstance(kv_pairs, CanAssignError):
            kv_pairs = [
                KVPair(
                    AnyValue(AnySource.generic_argument),
                    AnyValue(AnySource.generic_argument),
                )
            ]
        kv_pairs = list(reversed(kv_pairs))
        optional_pairs = set()
        removed_pairs = set()
        for key, pattern in zip(node.keys, node.patterns):
            key_val = self.visitor.visit(key)
            value, new_optional_pairs, new_removed_pairs = get_value_from_kv_pairs(
                kv_pairs, key_val, self.visitor, optional_pairs, removed_pairs
            )
            optional_pairs |= new_optional_pairs
            removed_pairs |= new_removed_pairs
            if value is UNINITIALIZED_VALUE:
                self.visitor.show_error(
                    node,
                    f"Impossible pattern: {self.visitor.match_subject.value} has no key"
                    f" {key_val}",
                    ErrorCode.impossible_pattern,
                )
                value = AnyValue(AnySource.error)
            with qcore.override(self.visitor, "match_subject", Composite(value)):
                constraints.append(self.visit(pattern))
        if node.rest is not None:
            new_kv_pairs = []
            for kv_pair in kv_pairs:
                if kv_pair in removed_pairs:
                    continue
                if kv_pair in optional_pairs:
                    kv_pair = replace(kv_pair, is_required=False)
                new_kv_pairs.append(kv_pair)
            val = DictIncompleteValue(dict, list(reversed(new_kv_pairs)))
            self.visitor._set_name_in_scope(node.rest, node, val)
        return AndConstraint.make(constraints)

    def visit_MatchClass(self, node: MatchClass) -> AbstractConstraint:
        cls = self.visitor.visit(node.cls)
        can_assign = TypedValue(type).can_assign(cls, self.visitor)
        if isinstance(can_assign, CanAssignError):
            self.visitor.show_error(
                node.cls,
                "Class pattern must be a type",
                ErrorCode.bad_match,
                detail=str(can_assign),
            )
        matched_type = type_from_value(cls, visitor=self.visitor, node=node.cls)
        self.check_impossible_pattern(node, matched_type)
        constraint = self.make_constraint(
            ConstraintType.predicate,
            # TODO figure out when to turn off positive_only
            IsAssignablePredicate(
                matched_type,
                self.visitor,
                positive_only=not node.patterns and not node.kwd_patterns,
            ),
        )
        subject = constrain_value(self.visitor.match_subject.value, constraint)
        subject_composite = self.visitor.match_subject._replace(value=subject)
        patterns = [
            (attr, pattern) for attr, pattern in zip(node.kwd_attrs, node.kwd_patterns)
        ]
        if node.patterns:
            match_args = get_match_args(cls, self.visitor)
            if isinstance(match_args, CanAssignError):
                self.visitor.show_error(
                    node.cls,
                    "Invalid class pattern",
                    ErrorCode.bad_match,
                    detail=str(match_args),
                )
                match_args = [SpecialPositionalMatch.error for _ in node.patterns]
            if len(node.patterns) > len(match_args):
                self.visitor.show_error(
                    node.cls,
                    f"{cls} takes at most {len(match_args)} positional subpatterns, but"
                    f" {len(match_args)} were provided",
                    ErrorCode.bad_match,
                    detail=str(match_args),
                )
                match_args = [SpecialPositionalMatch.error for _ in node.patterns]
            patterns = [*zip(match_args, node.patterns), *patterns]

        seen_names = set()
        for name, _ in patterns:
            if isinstance(name, str):
                if name in seen_names:
                    self.visitor.show_error(
                        node, f"Duplicate keyword pattern {name}", ErrorCode.bad_match
                    )
                seen_names.add(name)

        constraints = [constraint]
        for name, subpattern in patterns:
            if name is SpecialPositionalMatch.self:
                subsubject = subject_composite
            elif name is SpecialPositionalMatch.error:
                subsubject = Composite(AnyValue(AnySource.error))
            else:
                assert isinstance(name, str)
                attr = self.visitor.get_attribute(subject_composite, name)
                if attr is UNINITIALIZED_VALUE:
                    # It may exist on a child class, so we don't error here.
                    # This matches pyright's behavior.
                    subsubject = Composite(AnyValue(AnySource.unreachable))
                else:
                    new_varname = self.visitor._extend_composite(
                        subject_composite, name, subpattern
                    )
                    subsubject = Composite(attr, new_varname)
            with qcore.override(self.visitor, "match_subject", subsubject):
                constraints.append(self.visit(subpattern))

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


def index_of(elts: Sequence[T], pred: Callable[[T], bool]) -> Optional[int]:
    for i, elt in enumerate(elts):
        if pred(elt):
            return i
    return None


def get_value_from_kv_pairs(
    kv_pairs: Sequence[KVPair],
    key: Value,
    ctx: CanAssignContext,
    optional_pairs: Container[KVPair],
    removed_pairs: Container[KVPair],
) -> Tuple[Value, Set[KVPair], Set[KVPair]]:
    """Return the :class:`Value` for a specific key."""
    possible_values = []
    covered_keys: Set[Value] = set()
    new_optional_pairs: Set[KVPair] = set()
    for pair in kv_pairs:
        if pair in removed_pairs:
            continue
        if not pair.is_many:
            if isinstance(pair.key, AnnotatedValue):
                my_key = pair.key.value
            else:
                my_key = pair.key
            if isinstance(my_key, KnownValue):
                is_required = pair.is_required and pair not in optional_pairs
                if my_key == key and is_required:
                    if possible_values:
                        new_optional_pairs.add(pair)
                        new_removed_pairs = set()
                    else:
                        new_removed_pairs = {pair}
                    return (
                        unite_values(*possible_values, pair.value),
                        new_optional_pairs,
                        new_removed_pairs,
                    )
                elif my_key in covered_keys:
                    continue
                elif is_required:
                    covered_keys.add(my_key)
        if is_overlapping(key, pair.key, ctx):
            possible_values.append(pair.value)
            new_optional_pairs.add(pair)
    if not possible_values:
        return UNINITIALIZED_VALUE, set(), set()
    return unite_values(*possible_values), new_optional_pairs, set()


def get_match_args(
    cls: Value, visitor: "pyanalyze.name_check_visitor.NameCheckVisitor"
) -> Union[CanAssignError, Sequence[Union[str, SpecialPositionalMatch]]]:
    if SpecialClassPatternValue.is_assignable(cls, visitor):
        return [SpecialPositionalMatch.self]
    match_args_value = visitor.get_attribute(Composite(cls), "__match_args__")
    if match_args_value is UNINITIALIZED_VALUE:
        return CanAssignError(f"{cls} has no attribute __match_args__")
    match_args_value = replace_known_sequence_value(match_args_value)
    if (
        not isinstance(match_args_value, SequenceIncompleteValue)
        or match_args_value.typ is not tuple
    ):
        return CanAssignError(
            f"__match_args__ must be a literal tuple, not {match_args_value}"
        )
    match_args = []
    for i, arg in enumerate(match_args_value.members):
        if not isinstance(arg, KnownValue) or not isinstance(arg.val, str):
            return CanAssignError(
                f"__match_args__ element {i} is {arg}, not a string literal"
            )
        match_args.append(arg.val)
    return match_args
