"""

Visitor for pattern matching.

"""

import ast
from dataclasses import dataclass
import pyanalyze
from typing import Any

from .stacked_scopes import (
    NULL_CONSTRAINT,
    AbstractConstraint,
    Constraint,
    ConstraintType,
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


@dataclass
class PatmaVisitor(ast.NodeVisitor):
    visitor: "pyanalyze.name_check_visitor.NameCheckVisitor"

    def visit_MatchSingleton(self, node: MatchValue) -> AbstractConstraint:
        if self.visitor.match_subject.varname is None:
            return NULL_CONSTRAINT
        return Constraint(
            self.visitor.match_subject.varname,
            ConstraintType.is_value,
            True,
            node.value,
        )

    def generic_visit(self, node: ast.AST) -> AbstractConstraint:
        return NULL_CONSTRAINT
