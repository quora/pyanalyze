"""

Implementation of scope nesting in pyanalyze.

This module is responsible for mapping names to their values in pyanalyze. Variable lookup happens
mostly through a series of nested dictionaries. When pyanalyze sees a reference to a name inside a
nested function, it will first look at that function's scope, then in the enclosing function's
scope, then in the module scope, and finally in the builtin scope containing Python builtins. Each
of these scopes is represented as a :class:`Scope` object, which by default is just a thin
wrapper around a dictionary. However, function scopes are more complicated in order to track
variable values accurately through control flow structures like if blocks. See the
:class:`FunctionScope` docstring for details.

Other subtleties implemented here:

- Multiple assignments to the same name result in :class:`pyanalyze.value.MultiValuedValue`
- Globals are represented as :class:`pyanalyze.value.ReferencingValue`, and name lookups for such
  names are delegated to the :class:`pyanalyze.value.ReferencingValue`\'s scope
- Class scopes except the current one are skipped in name lookup

"""
import builtins
import contextlib
import enum
from ast import AST
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, replace
from itertools import chain
from types import ModuleType
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import qcore

from .boolability import get_boolability
from .extensions import reveal_type
from .safe import safe_equals, safe_issubclass
from .value import (
    annotate_value,
    AnnotatedValue,
    AnySource,
    AnyValue,
    ConstraintExtension,
    flatten_values,
    KnownValue,
    MultiValuedValue,
    NO_RETURN_VALUE,
    ReferencingValue,
    SubclassValue,
    TypedValue,
    TypeVarMap,
    UNINITIALIZED_VALUE,
    unite_and_simplify,
    unite_values,
    Value,
)

T = TypeVar("T")


LEAVES_SCOPE = "%LEAVES_SCOPE"
LEAVES_LOOP = "%LEAVES_LOOP"
_UNINITIALIZED = qcore.MarkerObject("uninitialized")


class VisitorState(enum.Enum):
    """The :term:`phase` of type checking."""

    collect_names = 1
    check_names = 2


class ScopeType(enum.Enum):
    builtin_scope = 1
    module_scope = 2
    class_scope = 3
    function_scope = 4


# Nodes as used in scopes can be any object, as long as they are hashable.
Node = object
# Tag for a Varname that changes when the variable is assigned to.
VarnameOrigin = FrozenSet[Optional[Node]]
CompositeIndex = Union[str, KnownValue]

EMPTY_ORIGIN = frozenset((None,))


@dataclass(frozen=True)
class CompositeVariable:
    """:term:`varname` used to implement constraints on instance variables.

    For example, access to ``self.x`` would make us use
    ``CompositeVariable('self', ('x',))``. If a function contains a check for
    ``isinstance(self.x, int)``, we would put a :class:`Constraint` on this
    :class:`CompositeVariable`.

    Also used for subscripts. Access to ``a[1]`` uses
    ``CompositeVariable('a', (KnownValue(1),))``. These can be mixed: ``a[1].b``
    corresponds to ``CompositeVariable('a', (KnownValue(1), 'b'))``.

    """

    varname: str
    attributes: Sequence[CompositeIndex]

    def __str__(self) -> str:
        pieces = [self.varname]
        for attr in self.attributes:
            if isinstance(attr, str):
                pieces.append(f".{attr}")
            else:
                pieces.append(f"[{attr.val!r}]")
        return "".join(pieces)


Varname = Union[str, CompositeVariable]


@dataclass(frozen=True)
class VarnameWithOrigin:
    varname: str
    origin: VarnameOrigin = EMPTY_ORIGIN
    indices: Sequence[Tuple[CompositeIndex, VarnameOrigin]] = ()

    def extend_with(
        self, index: CompositeIndex, origin: VarnameOrigin
    ) -> "VarnameWithOrigin":
        return VarnameWithOrigin(
            self.varname, self.origin, (*self.indices, (index, origin))
        )

    def get_all_varnames(self) -> Iterable[Tuple[Varname, VarnameOrigin]]:
        yield self.varname, self.origin
        for i, (_, origin) in enumerate(self.indices):
            varname = CompositeVariable(
                self.varname, tuple(index for index, _ in self.indices[: i + 1])
            )
            yield varname, origin

    def get_varname(self) -> Varname:
        if self.indices:
            return CompositeVariable(
                self.varname, tuple(index for index, _ in self.indices)
            )
        return self.varname

    def __str__(self) -> str:
        pieces = [self.varname]
        for index, _ in self.indices:
            if isinstance(index, str):
                pieces.append(f".{index}")
            else:
                pieces.append(f"[{index.val!r}]")
        return "".join(pieces)


SubScope = Dict[Varname, List[Node]]

# Type for Constraint.value if constraint type is predicate
# PredicateFunc = Callable[[Value, bool], Optional[Value]]


class Composite(NamedTuple):
    """A :class:`pyanalyze.value.Value` with information about its
    origin. This is useful for setting constraints."""

    value: Value
    varname: Optional[VarnameWithOrigin] = None
    node: Optional[AST] = None

    def get_extended_varname(self, index: CompositeIndex) -> Optional[Varname]:
        if self.varname is None:
            return None
        base = self.varname.get_varname()
        if isinstance(base, CompositeVariable):
            return CompositeVariable(base.varname, (*base.attributes, index))
        return CompositeVariable(base, (index,))

    def get_extended_varname_with_origin(
        self, index: CompositeIndex, origin: VarnameOrigin
    ) -> Optional[VarnameWithOrigin]:
        if self.varname is None:
            return None
        return self.varname.extend_with(index, origin)

    def substitute_typevars(self, typevars: TypeVarMap) -> "Composite":
        return Composite(
            self.value.substitute_typevars(typevars), self.varname, self.node
        )

    def __eq__(self, other: "Composite") -> bool:
        # Skip the AST node because it's hard to get right in tests.
        return (
            isinstance(other, Composite)
            and self.value == other.value
            and self.varname == other.varname
        )


@dataclass(frozen=True)
class _LookupContext:
    varname: Varname
    fallback_value: Optional[Value]
    node: Node
    state: VisitorState


class ConstraintType(enum.Enum):
    is_instance = 1
    """Corresponds to ``(not) isinstance(constraint.varname, constraint.value)``."""
    is_value = 2
    """Corresponds to ``constraint.varname is (not) constraint.value``."""
    is_truthy = 3
    """Corresponds to ``if (not) constraint.varname``."""
    one_of = 4
    """At least one of several other constraints on `varname` is true.

    For the `one_of` and `all_of` constraint types, the value is itself a list of constraints. These
    constraints are always positive. They are similar to the abstract
    :class:`AndConstraint` and :class:`OrConstraint`, but unlike these, all constraints in a
    `one_of` or `all_of` constraint apply to the same :term:`varname`.
    """
    all_of = 5
    """All of several other constraints on `varname` are true."""
    is_value_object = 6
    """`constraint.varname` should be typed as a :class:`pyanalyze.value.Value` object. Naming of
    this and `is_value` is confusing, and ideally we'd come up with better names."""
    predicate = 7
    """`constraint.value` is a `PredicateFunc`."""
    add_annotation = 8
    """`constraint.value` is an :class:`pyanalyze.value.Extension` to annotate the value with."""


class AbstractConstraint:
    """Base class for abstract constraints.

    We distinguish between abstract and concrete constraints. Abstract
    constraints are collected from conditions, and may be null constraints,
    concrete constraints, or an AND or OR of other abstract constraints.
    When we add constraints to a scope, we apply the abstract constraints to
    produce a set of concrete constraints. For example, a null constraint
    produces no concrete constraints, and an AND constraint AND(C1, C2)
    produces both C1 and C2.

    Concrete constraints are instances of the :class:`Constraint` class.

    """

    def apply(self) -> Iterable["Constraint"]:
        """Yields concrete constraints that are active when this constraint is applied.
        """
        raise NotImplementedError

    def invert(self) -> "AbstractConstraint":
        """Return an inverted version of this constraint."""
        raise NotImplementedError

    def __hash__(self) -> int:
        # Constraints need to be hashable by identity.
        return object.__hash__(self)


@dataclass(frozen=True, eq=False)
class Constraint(AbstractConstraint):
    """A constraint is a restriction on the value of a variable.

    Constraints are tracked in scope objects, so that we know which constraints
    are active for a given usage of a variable.

    For example::

        def f(x: Optional[int]) -> None:
            reveal_type(x)  # Union[int, None]
            assert x
            # Now a constraint of type is_truthy is active. Because
            # None is not truthy, we know that x is of type int.
            reveal_type(x)  # int

    """

    varname: VarnameWithOrigin
    """The :term:`varname` that the constraint applies to."""
    constraint_type: ConstraintType
    """Type of constraint. Determines the meaning of :attr:`value`."""
    positive: bool
    """Whether this is a positive constraint or not. For example,
    for an `is_truthy` constraint, ``if x`` would lead to a positive and ``if not x``
    to a negative constraint."""
    value: Any
    """Type for an ``is_instance`` constraint; value identical to the variable
    for ``is_value``; unused for is_truthy; :class:`pyanalyze.value.Value` object for
    `is_value_object`."""
    inverted: Optional["Constraint"] = field(
        compare=False, repr=False, hash=False, default=None
    )

    def __post_init__(self) -> None:
        assert isinstance(self.varname, VarnameWithOrigin), self.varname

    def apply(self) -> Iterable["Constraint"]:
        yield self

    def invert(self) -> "Constraint":
        if self.inverted is not None:
            return self.inverted
        inverted = Constraint(
            self.varname, self.constraint_type, not self.positive, self.value
        )
        object.__setattr__(self, "inverted", inverted)
        return inverted

    def apply_to_values(self, values: Iterable[Value]) -> Iterable[Value]:
        for value in values:
            yield from self.apply_to_value(value)

    def apply_to_value(self, value: Value) -> Iterable[Value]:
        """Yield values consistent with this constraint.

        Produces zero or more values consistent both with the given
        value and with this constraint.

        The value may not be a MultiValuedValue.

        """
        inner_value = value.value if isinstance(value, AnnotatedValue) else value
        if inner_value is UNINITIALIZED_VALUE:
            yield UNINITIALIZED_VALUE
            return
        if self.constraint_type == ConstraintType.is_instance:
            if isinstance(inner_value, AnyValue):
                if self.positive:
                    yield TypedValue(self.value)
                else:
                    yield inner_value
            elif isinstance(inner_value, KnownValue):
                if self.positive:
                    if isinstance(inner_value.val, self.value):
                        yield value
                else:
                    if not isinstance(inner_value.val, self.value):
                        yield value
            elif isinstance(inner_value, TypedValue):
                if isinstance(inner_value.typ, str):
                    # TODO handle synthetic types correctly here (which would require
                    # a CanAssignContext).
                    yield value
                elif self.positive:
                    if safe_issubclass(inner_value.typ, self.value):
                        yield value
                    elif safe_issubclass(self.value, inner_value.typ):
                        yield TypedValue(self.value)
                    # TODO: Technically here we should infer an intersection type:
                    # a type that is a subclass of both types. In practice currently
                    # _constrain_value() will eventually return NoReturn.
                else:
                    if not safe_issubclass(inner_value.typ, self.value):
                        yield value
            elif isinstance(inner_value, SubclassValue):
                if not isinstance(inner_value.typ, TypedValue):
                    yield value
                elif self.positive:
                    if isinstance(inner_value.typ.typ, self.value):
                        yield value
                else:
                    if not isinstance(inner_value.typ.typ, self.value):
                        yield value

        elif self.constraint_type == ConstraintType.is_value:
            if self.positive:
                known_val = KnownValue(self.value)
                if isinstance(inner_value, AnyValue):
                    yield known_val
                elif isinstance(inner_value, KnownValue):
                    if inner_value.val is self.value:
                        yield value
                elif isinstance(inner_value, TypedValue):
                    if isinstance(self.value, inner_value.typ):
                        yield known_val
                elif isinstance(inner_value, SubclassValue):
                    if (
                        isinstance(inner_value.typ, TypedValue)
                        and isinstance(self.value, type)
                        # TODO consider synthetic types
                        and isinstance(inner_value.typ.typ, type)
                        and safe_issubclass(self.value, inner_value.typ.typ)
                    ):
                        yield known_val
            else:
                if not (
                    isinstance(inner_value, KnownValue)
                    and inner_value.val is self.value
                ):
                    yield value

        elif self.constraint_type == ConstraintType.is_value_object:
            if self.positive:
                yield self.value
            else:
                # PEP 647 specifies that type narrowing should not happen
                # in the negative case.
                yield value

        elif self.constraint_type == ConstraintType.is_truthy:
            boolability = get_boolability(inner_value)
            if self.positive:
                if not boolability.is_safely_false():
                    yield value
            else:
                if not boolability.is_safely_true():
                    yield value

        elif self.constraint_type == ConstraintType.predicate:
            new_value = self.value(value, self.positive)
            if new_value is not None:
                yield new_value

        elif self.constraint_type == ConstraintType.add_annotation:
            if self.positive:
                yield annotate_value(value, [self.value])
            else:
                yield value

        elif self.constraint_type == ConstraintType.one_of:
            for constraint in self.value:
                yield from constraint.apply_to_value(value)

        elif self.constraint_type == ConstraintType.all_of:
            vals = [value]
            for constraint in self.value:
                vals = list(constraint.apply_to_values(vals))
            yield from vals

        else:
            assert False, f"unknown constraint type {self.constraint_type}"

    def __str__(self) -> str:
        sign = "+" if self.positive else "-"
        if isinstance(self.value, list):
            value = str(list(map(str, self.value)))
        else:
            value = str(self.value)
        return f"<{sign}{self.varname} {self.constraint_type.name} {value}>"


TRUTHY_CONSTRAINT = Constraint(
    VarnameWithOrigin("%unused"), ConstraintType.is_truthy, True, None
)
FALSY_CONSTRAINT = Constraint(
    VarnameWithOrigin("%unused"), ConstraintType.is_truthy, False, None
)


@dataclass(frozen=True)
class NullConstraint(AbstractConstraint):
    """Represents the absence of a constraint."""

    def apply(self) -> Iterable[Constraint]:
        return []

    def invert(self) -> "NullConstraint":
        return self


NULL_CONSTRAINT = NullConstraint()
"""The single instance of :class:`NullConstraint`."""


@dataclass(frozen=True)
class PredicateProvider(AbstractConstraint):
    """A form of constraint implemented through a predicate on a value.

    If a function returns a :class:`PredicateProvider`, equality
    checks on the return value will produce a `predicate`
    :term:`constraint`.

    Consider the following code::

        def two_lengths(tpl: Union[Tuple[int], Tuple[str, int]]) -> int:
            if len(tpl) == 1:
                return tpl[0]
            else:
                return tpl[1]

    The :term:`impl` for :func:`len` returns a :class:`PredicateProvider`,
    with a `provider` attribute that returns the length of the object
    represented by a :term:`value`. In turn, the equality check (``== 1``)
    produces a constraint of type `predicate`, which filters away any
    values that do not match the length of the object.

    In this case, there are two values: a tuple of length 1 and one of
    length 2. Only the first matches the constraint, so the type is
    narrowed down to that tuple and the code typechecks correctly.

    """

    varname: VarnameWithOrigin
    provider: Callable[[Value], Value]

    def apply(self) -> Iterable[Constraint]:
        return []

    def invert(self) -> AbstractConstraint:
        # inverting is meaningless
        return NULL_CONSTRAINT


@dataclass(frozen=True)
class EquivalentConstraint(AbstractConstraint):
    """Represents multiple constraints that are either all true or all false."""

    constraints: Tuple[AbstractConstraint, ...]

    def apply(self) -> Iterable["Constraint"]:
        for cons in self.constraints:
            yield from cons.apply()

    def invert(self) -> "EquivalentConstraint":
        # ~(A == B) -> ~A == ~B
        return EquivalentConstraint(tuple(cons.invert() for cons in self.constraints))

    @classmethod
    def make(cls, constraints: Iterable[AbstractConstraint]) -> AbstractConstraint:
        processed = {}
        for cons in constraints:
            if isinstance(cons, EquivalentConstraint):
                for subcons in cons.constraints:
                    processed[id(subcons)] = subcons
                continue
            processed[id(cons)] = cons

        final = list(processed.values())

        if len(final) == 1:
            (cons,) = final
            return cons
        return cls(tuple(final))

    def __str__(self) -> str:
        children = " == ".join(map(str, self.constraints))
        return f"({children})"


@dataclass(frozen=True)
class AndConstraint(AbstractConstraint):
    """Represents the AND of two constraints."""

    constraints: Tuple[AbstractConstraint, ...]

    def apply(self) -> Iterable["Constraint"]:
        for cons in self.constraints:
            yield from cons.apply()

    def invert(self) -> "OrConstraint":
        # ~(A and B) -> ~A or ~B
        return OrConstraint(tuple(cons.invert() for cons in self.constraints))

    @classmethod
    def make(cls, constraints: Iterable[AbstractConstraint]) -> AbstractConstraint:
        processed = {}
        for cons in constraints:
            if isinstance(cons, AndConstraint):
                for subcons in cons.constraints:
                    processed[id(subcons)] = subcons
                continue
            processed[id(cons)] = cons

        final = []
        for constraint in processed.values():
            if isinstance(constraint, OrConstraint):
                # A AND (A OR B) reduces to a
                if any(id(subcons) in processed for subcons in constraint.constraints):
                    continue
            final.append(constraint)

        if not final:
            return NULL_CONSTRAINT
        if len(final) == 1:
            (cons,) = final
            return cons
        return cls(tuple(final))

    def __str__(self) -> str:
        children = " AND ".join(map(str, self.constraints))
        return f"({children})"


@dataclass(frozen=True)
class OrConstraint(AbstractConstraint):
    """Represents the OR of two constraints."""

    constraints: Tuple[AbstractConstraint, ...]

    def apply(self) -> Iterable[Constraint]:
        grouped = [self._group_constraints(cons) for cons in self.constraints]
        left, *rest = grouped
        for varname, constraints in left.items():
            # Produce one_of constraints if the same variable name
            # applies on both the left and the right side.
            if all(varname in group for group in rest):
                constraints = [
                    self._constraint_from_list(varname, constraints),
                    *[
                        self._constraint_from_list(varname, group[varname])
                        for group in rest
                    ],
                ]
                yield Constraint(
                    varname, ConstraintType.one_of, True, list(set(constraints))
                )

    def _constraint_from_list(
        self, varname: VarnameWithOrigin, constraints: Sequence[Constraint]
    ) -> Constraint:
        if len(constraints) == 1:
            return constraints[0]
        else:
            return Constraint(varname, ConstraintType.all_of, True, constraints)

    def _group_constraints(
        self, abstract_constraint: AbstractConstraint
    ) -> Dict[VarnameWithOrigin, List[Constraint]]:
        by_varname = defaultdict(list)
        for constraint in abstract_constraint.apply():
            by_varname[constraint.varname].append(constraint)
        return by_varname

    def invert(self) -> AndConstraint:
        # ~(A or B) -> ~A and ~B
        return AndConstraint(tuple(cons.invert() for cons in self.constraints))

    @classmethod
    def make(cls, constraints: Iterable[AbstractConstraint]) -> AbstractConstraint:
        processed = {}
        for cons in constraints:
            if isinstance(cons, OrConstraint):
                for subcons in cons.constraints:
                    processed[id(subcons)] = subcons
                continue
            processed[id(cons)] = cons

        final = []
        for constraint in processed.values():
            if isinstance(constraint, AndConstraint):
                # A OR (A AND B) reduces to a
                if any(id(subcons) in processed for subcons in constraint.constraints):
                    continue
            elif isinstance(constraint, Constraint):
                inverted = id(constraint.invert())
                if inverted in processed:
                    continue
            final.append(constraint)

        if not final:
            return NULL_CONSTRAINT
        if len(final) == 1:
            (cons,) = final
            return cons
        return cls(tuple(final))

    def __str__(self) -> str:
        children = " OR ".join(map(str, self.constraints))
        return f"({children})"


@dataclass(frozen=True)
class _ConstrainedValue(Value):
    """Helper class, only used within a FunctionScope."""

    definition_nodes: FrozenSet[Node]
    constraints: Sequence[Constraint]
    resolution_cache: Dict[_LookupContext, Value] = field(
        default_factory=dict, init=False, compare=False, hash=False, repr=False
    )


_empty_constrained = _ConstrainedValue(frozenset(), [])


@dataclass
class Scope:
    """Represents a single level in the scope stack.

    May be a builtin, module, class, or function scope.

    """

    scope_type: ScopeType
    variables: Dict[Varname, Value] = field(default_factory=dict)
    parent_scope: Optional["Scope"] = None
    scope_node: Optional[Node] = None
    scope_object: Optional[object] = None
    simplification_limit: Optional[int] = None
    declared_types: Dict[str, Tuple[Optional[Value], bool, AST]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.parent_scope is not None:
            self.parent_scope = self.parent_scope.scope_used_as_parent()

    def add_constraint(
        self, abstract_constraint: AbstractConstraint, node: Node, state: VisitorState
    ) -> None:
        """Constraints are ignored outside of function scopes."""
        pass

    def get(
        self,
        varname: Varname,
        node: object,
        state: VisitorState,
        from_parent_scope: bool = False,
    ) -> Tuple[Value, Optional["Scope"], VarnameOrigin]:
        local_value, origin = self.get_local(
            varname, node, state, from_parent_scope=from_parent_scope
        )
        if local_value is not UNINITIALIZED_VALUE:
            return self.resolve_reference(local_value, state), self, origin
        elif self.parent_scope is not None:
            # Parent scopes don't get the node to help local lookup.
            parent_node = (
                (varname, self.scope_node) if self.scope_node is not None else None
            )
            val, scope, _ = self.parent_scope.get(
                varname, parent_node, state, from_parent_scope=True
            )
            # Tag lookups in the parent scope with this scope node, so we
            # don't carry over constraints across scopes.
            return val, scope, EMPTY_ORIGIN
        else:
            return UNINITIALIZED_VALUE, None, EMPTY_ORIGIN

    def get_local(
        self,
        varname: Varname,
        node: Node,
        state: VisitorState,
        from_parent_scope: bool = False,
        fallback_value: Optional[Value] = None,
    ) -> Tuple[Value, VarnameOrigin]:
        if varname in self.variables:
            return self.variables[varname], EMPTY_ORIGIN
        else:
            return UNINITIALIZED_VALUE, EMPTY_ORIGIN

    def get_origin(
        self, varname: Varname, node: Node, state: VisitorState
    ) -> VarnameOrigin:
        return EMPTY_ORIGIN

    def set(
        self, varname: Varname, value: Value, node: Node, state: VisitorState
    ) -> VarnameOrigin:
        if varname not in self.variables:
            self.variables[varname] = value
        elif isinstance(value, AnyValue) or not safe_equals(
            self.variables[varname], value
        ):
            existing = self.variables[varname]
            if isinstance(existing, ReferencingValue):
                existing.scope.set(existing.name, value, node, state)
            elif (
                type(existing) is TypedValue
                and isinstance(value, TypedValue)
                # TODO constraints for type(...) is
                # static analysis: ignore[attribute_is_never_set]
                and existing.typ is value.typ
            ):
                # replace with a more specific TypedValue
                self.variables[varname] = value
            else:
                self.variables[varname] = unite_values(existing, value)
        return EMPTY_ORIGIN

    def items(self) -> Iterable[Tuple[Varname, Value]]:
        return self.variables.items()

    def all_variables(self) -> Iterable[Varname]:
        return self.variables

    def set_declared_type(
        self, varname: str, typ: Optional[Value], is_final: bool, node: AST
    ) -> bool:
        if varname in self.declared_types:
            _, _, existing_node = self.declared_types[varname]
            already_present = node is not existing_node
            # Don't replace the existing node, or we'll generate spurious already_declared
            # errors.
            node = existing_node
        else:
            already_present = False
        # Even if we give an error, still honor the later type.
        self.declared_types[varname] = (typ, is_final, node)
        return not already_present

    def get_declared_type(self, varname: str) -> Optional[Value]:
        if varname not in self.declared_types:
            return None
        typ, _, _ = self.declared_types[varname]
        return typ

    def is_final(self, varname: str) -> bool:
        if varname not in self.declared_types:
            return False
        _, is_final, _ = self.declared_types[varname]
        return is_final

    def __contains__(self, varname: Varname) -> bool:
        return varname in self.variables or varname in self.declared_types

    @contextlib.contextmanager
    def suppressing_subscope(self) -> Iterator[SubScope]:
        yield {}

    # no real subscopes in non-function scopes, just dummy implementations
    @contextlib.contextmanager
    def subscope(self) -> Iterator[None]:
        yield

    @contextlib.contextmanager
    def loop_scope(self) -> Iterator[None]:
        # Context manager for the subscope associated with a loop.
        yield

    def combine_subscopes(
        self, scopes: Iterable[SubScope], *, ignore_leaves_scope: bool = False
    ) -> None:
        pass

    def resolve_reference(self, value: Value, state: VisitorState) -> Value:
        if isinstance(value, ReferencingValue):
            referenced, _, _ = value.scope.get(value.name, None, state)
            # globals that are None are probably set to something else later
            if safe_equals(referenced, KnownValue(None)):
                return AnyValue(AnySource.inference)
            else:
                return referenced
        else:
            return value

    def scope_used_as_parent(self) -> "Scope":
        """Class scopes are skipped in scope lookup, so don't set them as parent scopes.
        """
        if self.scope_type == ScopeType.class_scope:
            assert (
                self.parent_scope is not None
            ), "class scopes must have a parent scope"
            return self.parent_scope.scope_used_as_parent()
        else:
            return self


class FunctionScope(Scope):
    """Keeps track of the local variables of a single function.

    :class:`FunctionScope` is designed to produce the correct value for each variable at each point
    in the function, unlike the base :class:`Scope` class, which assumes that each variable has
    the same value throughout the scope it represents.

    For example, given the code::

        x = 3
        x = 4
        print(x)

    :class:`FunctionScope` will infer the value of `x` to be ``KnownValue(4)``, but :class:`Scope`
    will produce a :class:`pyanalyze.value.MultiValuedValue` because it does not know whether the
    assignment to 3 or 4 is active.

    The approach taken is to map each usage node (a place where the variable is used) to a set of
    definition nodes (places where the variable is assigned to) that could be active when the
    variable is used. Each definition node is also mapped to the value assigned to the variable
    there.

    For example, in the code::

        x = 3  # (a)
        print(x)  # (b)

    (a) is the only definition node for the usage node at (b), and (a) is mapped to
    ``KnownValue(3)``, so at (b) x is inferred to be ``KnownValue(3)``.

    However, in this code::

        if some_condition():
            x = 3  # (a)
        else:
            x = 4  # (b)
        print(x)  # (c)

    both (a) and (b) are possible definition nodes for the usage node at (c), so at (c) x is
    inferred to be a ``MultiValuedValue([KnownValue(3), KnownValue(4)])``.

    These mappings are implemented as the `usage_to_definition_nodes` and `definition_node_to_value`
    attributes on the :class:`FunctionScope` object. They are created completely during the
    collecting :term:`phase`. The basic mechanism uses the `name_to_current_definition_nodes`
    dictionary, which maps each local variable to a list of active definition nodes. When pyanalyze
    encounters an assignment, it updates `name_to_current_definition_nodes` to map to that
    assignment node, and when it encounters a variable usage it updates `usage_to_definition_nodes`
    to map that usage to the current definition nodes in `name_to_current_definition_nodes`. For
    example::

        # name_to_current_definition_nodes (n2cdn) = {}, usage_to_definition_nodes (u2dn) = {}
        x = 3  # (a)
        # n2cdn = {'x': [(a)]}, u2dn = {}
        print(x)  # (b)
        # n2cdn = {'x': [(a)]}, u2dn = {(b): [(a)]}
        x = 4  # (c)
        # n2cdn = {'x': [(c)]}, u2dn = {(b): [(a)]}
        print(x)  # (d)
        # n2cdn = {'x': [(c)]}, u2dn = {(b): [(a)], (d): [(c)]}

    However, this simple approach is not sufficient to handle control flow inside the function. To
    handle this case, :class:`FunctionScope` supports the creation of subscopes and the
    `combine_subscopes` operation. Each branch in a conditional statement is mapped to a separate
    subscope, which contains an independently updated copy of `name_to_current_definition_nodes`.
    After pyanalyze visits all branches, it runs the `combine_subscopes` operation on all of the
    branches' subscopes. This operation takes, for each variable, the union of the definition nodes
    created in all of the branches. For example::

        # n2cdn = {}, u2dn = {}
        if some_condition():
            # subscope 1
            x = 3  # (a)
            print(x)  # (b)
            # n2cdn = {'x': [(a)]}, u2dn = {(b): [(a)]}
        else:
            # subscope 2
            x = 4  # (c)
            print(x)  # (d)
            # n2cdn = {'x': [(c)]}, u2dn = {(b): [(a)], (d): [(c)]}
        # combine_subscopes([subscope 1, subscope 2]) happens
        # n2cdn = {'x': [(a), (c)]}, u2dn = {(b): [(a)], (d): [(c)]}
        print(x)  # (e)
        # n2cdn = {'x': [(a), (c)]}, u2dn = {(b): [(a)], (d): [(c)], (e): [(a), (c)]}

    This model applies most cleanly to if blocks, but try-except can also be analyzed using this
    approach. Loops are more complicated, because variable usages sometimes need to be mapped to
    definition nodes later in the same loop body. For example, in code like this::

        x = None
        for _ in (1, 2):
            if x:
                print(x[1])  # (a)
            else:
                x = (1, 2)  # (b)

    a naive approach would infer that `x` is ``None`` at (a). To take care of this case, pyanalyze
    visits the loop body twice during the collecting :term:`phase`, so that
    `usage_to_definition_nodes` can add a mapping of (a) to (b). To handle `break` and `continue`
    correctly, it also uses a separate "loop scope" that ends up combining the scopes created by
    normal control flow through the body of the loop and by each `break` and `continue` statement.

    Try-finally blocks are handled by visiting the finally block twice. Essentially, we treat::

        try:
            TRY-BODY
        finally:
            FINALLY-BODY
        REST-OF-FUNCTION

    as equivalent to::

        if <empty>:
            FINALLY-BODY
            return
        else:
            TRY-BODY
            FINALLY-BODY
            REST-OF-FUNCTION

    This correctly expresses that variables used in the FINALLY-BODY can have either the values set
    in the TRY-BODY or those set before the try-finally. It does not express that the TRY-BODY may
    have been interrupted at any point, but that does not matter for our purposes. It has the
    disadvantage that the finally body is visted twice, which may lead to some errors being doubled.

    A similar approach is used to handle loops, where the body of the loop may not be executed at
    all. A for loop of the form::

        for TARGET in ITERABLE:
            FOR-BODY
        else:
            ELSE-BODY

    is treated like::

        if <empty>:
            TARGET = next(iter(ITERABLE))
            FOR-BODY
        else:
            ELSE_BODY

    Special logic is also needed to take care of globals (which are kept track of separately from
    normal variables) and variable lookups without a node context. For the latter, the
    name_to_all_definition_nodes maps each variable name to all possible definition nodes.

    """

    name_to_current_definition_nodes: SubScope
    usage_to_definition_nodes: Dict[Tuple[Node, Varname], List[Node]]
    definition_node_to_value: Dict[Node, Value]
    name_to_all_definition_nodes: Dict[Varname, Set[Node]]
    name_to_composites: Dict[Varname, Set[CompositeVariable]]
    referencing_value_vars: Dict[Varname, Value]
    accessed_from_special_nodes: Set[Varname]
    current_loop_scopes: List[SubScope]

    def __init__(
        self,
        parent_scope: Scope,
        scope_node: Optional[Node] = None,
        simplification_limit: Optional[int] = None,
    ) -> None:
        super().__init__(
            ScopeType.function_scope,
            {},
            parent_scope,
            scope_node,
            simplification_limit=simplification_limit,
        )
        self.name_to_current_definition_nodes = defaultdict(list)
        self.usage_to_definition_nodes = defaultdict(list)
        self.definition_node_to_value = {_UNINITIALIZED: _empty_constrained}
        self.name_to_all_definition_nodes = defaultdict(set)
        self.name_to_composites = defaultdict(set)
        self.referencing_value_vars = defaultdict(lambda: UNINITIALIZED_VALUE)
        # Names that are accessed from a None node context (e.g., from a nested function). These
        # are ignored when looking at unused variables.
        self.accessed_from_special_nodes = set()
        self.current_loop_scopes = []

    def add_constraint(
        self, abstract_constraint: AbstractConstraint, node: Node, state: VisitorState
    ) -> None:
        """Add a new constraint.

        Constraints are represented as assignments of fake values, which are
        _ConstrainedValue objects. These contain a constraint and a set of definition
        nodes where the unconstrained variable could have been defined. When we try to retrieve
        the value of a variable, we look at the values in each of the definition node and
        at the constraint.

        The node argument may be any unique key, although it will usually be an AST node.

        """
        for constraint in abstract_constraint.apply():
            self._add_single_constraint(constraint, node, state)

    def _add_single_constraint(
        self, constraint: Constraint, node: Node, state: VisitorState
    ) -> None:
        for parent_varname, constraint_origin in constraint.varname.get_all_varnames():
            current_origin = self.get_origin(parent_varname, node, state)
            current_set = self._resolve_origin(current_origin)
            constraint_set = self._resolve_origin(constraint_origin)
            if current_set - constraint_set:
                return

        varname = constraint.varname.get_varname()
        def_nodes = frozenset(self.name_to_current_definition_nodes[varname])
        # We set both a constraint and its inverse using the same node as the definition
        # node, so cheat and include the constraint itself in the key.
        node = (node, constraint)
        val = _ConstrainedValue(def_nodes, [constraint])
        self.definition_node_to_value[node] = val
        self.name_to_current_definition_nodes[varname] = [node]
        self._add_composite(varname)

    def _resolve_origin(self, definers: Iterable[Node]) -> FrozenSet[Node]:
        seen = set()
        pending = set(definers)
        out = set()
        while pending:
            definer = pending.pop()
            if definer in seen:
                continue
            seen.add(definer)
            if definer is None:
                out.add(None)
            elif definer not in self.definition_node_to_value:
                # maybe from a different scope
                return EMPTY_ORIGIN
            else:
                val = self.definition_node_to_value[definer]
                if isinstance(val, _ConstrainedValue):
                    pending |= val.definition_nodes
                else:
                    out.add(definer)
        if not out:
            return EMPTY_ORIGIN
        return frozenset(out)

    def set(
        self, varname: Varname, value: Value, node: Node, state: VisitorState
    ) -> VarnameOrigin:
        if isinstance(value, ReferencingValue):
            self.referencing_value_vars[varname] = value
            return EMPTY_ORIGIN
        ref_var = self.referencing_value_vars[varname]
        if isinstance(ref_var, ReferencingValue):
            ref_var.scope.set(ref_var.name, value, node, state)
            # Mark anything set through a ReferencingValue as special access; this disables unused
            # variable errors. We can't reliably find out that a nonlocal variable is unused.
            if isinstance(ref_var.scope, FunctionScope):
                ref_var.scope.accessed_from_special_nodes.add(varname)
            self.accessed_from_special_nodes.add(varname)
        self.definition_node_to_value[node] = value
        self.name_to_current_definition_nodes[varname] = [node]
        for composite in self.name_to_composites[varname]:
            # After we assign to a variable, reset any constraints on its
            # members.
            self.name_to_current_definition_nodes[composite] = []
        self.name_to_all_definition_nodes[varname].add(node)
        self._add_composite(varname)
        return frozenset([node])

    def get_local(
        self,
        varname: Varname,
        node: Node,
        state: VisitorState,
        from_parent_scope: bool = False,
        fallback_value: Optional[Value] = None,
    ) -> Tuple[Value, VarnameOrigin]:
        self._add_composite(varname)
        ctx = _LookupContext(varname, fallback_value, node, state)
        if from_parent_scope:
            self.accessed_from_special_nodes.add(varname)
        key = (node, varname)
        if node is None:
            self.accessed_from_special_nodes.add(varname)
            # this indicates that we're not looking at a normal local variable reference, but
            # something special like a nested function
            if varname in self.name_to_all_definition_nodes:
                definers = self.name_to_all_definition_nodes[varname]
            else:
                return self.referencing_value_vars[varname], EMPTY_ORIGIN
        elif state is VisitorState.check_names:
            if key not in self.usage_to_definition_nodes:
                return self.referencing_value_vars[varname], EMPTY_ORIGIN
            else:
                definers = self.usage_to_definition_nodes[key]
        else:
            if varname in self.name_to_current_definition_nodes:
                definers = self.name_to_current_definition_nodes[varname]
                self.usage_to_definition_nodes[key] += definers
            else:
                return self.referencing_value_vars[varname], EMPTY_ORIGIN
        return self._get_value_from_nodes(definers, ctx), self._resolve_origin(definers)

    def get_origin(
        self, varname: Varname, node: Node, state: VisitorState
    ) -> VarnameOrigin:
        key = (node, varname)
        if node is None:
            # this indicates that we're not looking at a normal local variable reference, but
            # something special like a nested function
            if varname in self.name_to_all_definition_nodes:
                definers = self.name_to_all_definition_nodes[varname]
            else:
                return EMPTY_ORIGIN
        elif state is VisitorState.check_names:
            if key not in self.usage_to_definition_nodes:
                return EMPTY_ORIGIN
            else:
                definers = self.usage_to_definition_nodes[key]
        else:
            if varname in self.name_to_current_definition_nodes:
                definers = self.name_to_current_definition_nodes[varname]
                self.usage_to_definition_nodes[key] += definers
            else:
                return EMPTY_ORIGIN
        return self._resolve_origin(definers)

    def get_all_definition_nodes(self) -> Dict[Varname, Set[Node]]:
        """Return a copy of name_to_all_definition_nodes."""
        return {
            key: set(nodes) for key, nodes in self.name_to_all_definition_nodes.items()
        }

    @contextlib.contextmanager
    def suppressing_subscope(self) -> Iterator[SubScope]:
        """A suppressing subscope is a subscope that may suppress exceptions
        inside of it.

        This is used to implement try and with blocks. After code like this::

            x = 1
            try:
                x = 2
                x = 3
            except Exception:
                pass

        The value of `x` may be any of 1, 2, and 3, depending on whether and
        where an exception was thrown.

        To implement this, we keep track of all assignments inside the block
        and give them effect, so that after the suppressing subscope ends,
        each variable's definition nodes include all of these assignments.

        """
        old_defn_nodes = self.get_all_definition_nodes()
        with self.subscope() as inner_scope:
            yield inner_scope
        new_defn_nodes = self.get_all_definition_nodes()
        rest_scope = {
            key: list(nodes - old_defn_nodes.get(key, set()))
            for key, nodes in new_defn_nodes.items()
            if key != LEAVES_SCOPE
        }
        rest_scope = {key: nodes for key, nodes in rest_scope.items() if nodes}
        with self.subscope() as dummy_subscope:
            pass
        all_keys = set(rest_scope) | set(dummy_subscope)
        new_scope = {
            key: [*dummy_subscope.get(key, []), *rest_scope.get(key, [])]
            for key in all_keys
        }
        self.combine_subscopes([dummy_subscope, new_scope])

    @contextlib.contextmanager
    def subscope(self) -> Iterator[SubScope]:
        """Create a new subscope, to be used for conditional branches."""
        # Ignore LEAVES_SCOPE if it's already there, so that we type check code after the
        # assert False correctly. Without this, test_after_assert_false fails.
        new_name_to_nodes = defaultdict(
            list,
            {
                key: value
                for key, value in self.name_to_current_definition_nodes.items()
                if key != LEAVES_SCOPE
            },
        )
        with qcore.override(
            self, "name_to_current_definition_nodes", new_name_to_nodes
        ):
            yield new_name_to_nodes

    @contextlib.contextmanager
    def loop_scope(self) -> Iterator[List[SubScope]]:
        loop_scopes = []
        with self.subscope() as main_scope:
            loop_scopes.append(main_scope)
            with qcore.override(self, "current_loop_scopes", loop_scopes):
                yield loop_scopes
        self.combine_subscopes(
            [
                {name: values for name, values in scope.items() if name != LEAVES_LOOP}
                for scope in loop_scopes
            ]
        )

    def get_combined_scope(
        self, scopes: Iterable[SubScope], *, ignore_leaves_scope: bool = False
    ) -> SubScope:
        new_scopes = []
        for scope in scopes:
            if LEAVES_LOOP in scope:
                self.current_loop_scopes.append(scope)
            elif LEAVES_SCOPE not in scope or ignore_leaves_scope:
                new_scopes.append(scope)
        if not new_scopes:
            return {LEAVES_SCOPE: []}
        all_variables = set(chain.from_iterable(new_scopes))
        return {
            varname: uniq_chain(
                scope.get(varname, [_UNINITIALIZED]) for scope in new_scopes
            )
            for varname in all_variables
        }

    def combine_subscopes(
        self, scopes: Iterable[SubScope], *, ignore_leaves_scope: bool = False
    ) -> None:
        self.name_to_current_definition_nodes.update(
            self.get_combined_scope(scopes, ignore_leaves_scope=ignore_leaves_scope)
        )

    def _resolve_value(self, val: Value, ctx: _LookupContext) -> Value:
        if isinstance(val, _ConstrainedValue):
            # Cache repeated resolutions of the same ConstrainedValue, because otherwise
            # lots of nested constraints can lead to exponential performance (see the
            # test_repeated_constraints test case).
            key = replace(ctx, fallback_value=None)
            if key in val.resolution_cache:
                return val.resolution_cache[key]
            # Guard against recursion. This happens in the test_len_condition test.
            # Perhaps we should do something smarter to prevent recursion.
            val.resolution_cache[key] = NO_RETURN_VALUE
            if val.definition_nodes or ctx.fallback_value:
                resolved = self._get_value_from_nodes(
                    val.definition_nodes, ctx, val.constraints
                )
            else:
                assert (
                    self.parent_scope
                ), "constrained value must have definition nodes or parent scope"
                parent_val, _, _ = self.parent_scope.get(ctx.varname, None, ctx.state)
                resolved = _constrain_value(
                    [parent_val],
                    val.constraints,
                    simplification_limit=self.simplification_limit,
                )
            val.resolution_cache[key] = resolved
            return resolved
        else:
            return val

    def _get_value_from_nodes(
        self,
        nodes: Iterable[Node],
        ctx: _LookupContext,
        constraints: Iterable[Constraint] = (),
    ) -> Value:
        # Deduplicate nodes to gain some performance.
        nodes = OrderedDict.fromkeys(nodes)
        # If the variable is a nonlocal or composite, "uninitialized" doesn't make sense;
        # instead use an empty constraint to point to the parent scope.
        should_use_unconstrained = (
            isinstance(ctx.varname, CompositeVariable)
            or (ctx.varname not in self.name_to_all_definition_nodes)
            or isinstance(self.referencing_value_vars[ctx.varname], ReferencingValue)
        )
        values = [
            UNINITIALIZED_VALUE
            if node is _UNINITIALIZED and not should_use_unconstrained
            else self._resolve_value(self.definition_node_to_value[node], ctx)
            for node in nodes
        ]
        return _constrain_value(
            values,
            constraints,
            fallback_value=ctx.fallback_value,
            simplification_limit=self.simplification_limit,
        )

    def _add_composite(self, varname: Varname) -> None:
        if isinstance(varname, CompositeVariable):
            self.name_to_composites[varname.varname].add(varname)
            if len(varname.attributes) > 1:
                for i in range(1, len(varname.attributes)):
                    composite = CompositeVariable(
                        varname.varname, varname.attributes[:i]
                    )
                    self.name_to_composites[composite].add(varname)

    def items(self) -> Iterable[Tuple[Varname, Value]]:
        raise NotImplementedError

    def all_variables(self) -> Iterable[Varname]:
        yield from self.name_to_current_definition_nodes

    def __contains__(self, varname: Varname) -> bool:
        return (
            varname in self.name_to_all_definition_nodes
            or varname in self.declared_types
        )


class StackedScopes:
    """Represents the stack of scopes in which Python searches for variables."""

    _builtin_scope = Scope(
        ScopeType.builtin_scope,
        {
            **{k: KnownValue(v) for k, v in builtins.__dict__.items()},
            "reveal_type": KnownValue(reveal_type),
        },
        None,
    )

    def __init__(
        self,
        module_vars: Dict[str, Value],
        module: Optional[ModuleType],
        *,
        simplification_limit: Optional[int] = None,
    ) -> None:
        self.simplification_limit = simplification_limit
        self.scopes = [
            self._builtin_scope,
            Scope(
                ScopeType.module_scope,
                module_vars,
                self._builtin_scope,
                scope_object=module,
                simplification_limit=simplification_limit,
            ),
        ]

    @contextlib.contextmanager
    def add_scope(
        self,
        scope_type: ScopeType,
        scope_node: Node,
        scope_object: Optional[object] = None,
    ) -> Iterator[None]:
        """Context manager that adds a scope of this type to the top of the stack."""
        if scope_type is ScopeType.function_scope:
            scope = FunctionScope(
                self.scopes[-1], scope_node, self.simplification_limit
            )
        else:
            scope = Scope(
                scope_type,
                {},
                self.scopes[-1],
                scope_node,
                scope_object=scope_object,
                simplification_limit=self.simplification_limit,
            )
        self.scopes.append(scope)
        try:
            yield
        finally:
            self.scopes.pop()

    @contextlib.contextmanager
    def ignore_topmost_scope(self) -> Iterator[None]:
        """Context manager that temporarily ignores the topmost scope."""
        scope = self.scopes.pop()
        try:
            yield
        finally:
            self.scopes.append(scope)

    @contextlib.contextmanager
    def allow_only_module_scope(self) -> Iterator[None]:
        """Context manager that allows only lookups in the module and builtin scopes."""
        rest = self.scopes[2:]
        del self.scopes[2:]
        try:
            yield
        finally:
            self.scopes += rest

    def get(self, varname: Varname, node: Node, state: VisitorState) -> Value:
        """Gets a variable of the given name from the current scope stack.

        :param varname: :term:`varname` of the variable to retrieve
        :type varname: Varname

        :param node: AST node corresponding to the place where the variable lookup is happening.
                     :class:`FunctionScope` uses this to decide which definition of the variable
                     to use; other scopes ignore it. It can be passed as None to indicate that
                     any definition may be used. This is used among others when looking up names
                     in outer scopes. Although this argument should normally be an AST node, it
                     can be any unique, hashable identifier, because sometimes a single AST node
                     sets multiple variables (e.g. in ImportFrom nodes).
        :type node: Node

        :param state: The current :class:`VisitorState`. Pyanalyze runs the collecting
                      :term:`phase` to collect all name assignments and map name usages to their
                      corresponding assignments, and then the checking phase to locate any errors
                      in the code.
        :type state: VisitorState

        Returns :data:`pyanalyze.value.UNINITIALIZED_VALUE` if the name is not defined in any known
        scope.

        """
        value, _, _ = self.get_with_scope(varname, node, state)
        return value

    def get_with_scope(
        self, varname: Varname, node: Node, state: VisitorState
    ) -> Tuple[Value, Optional[Scope], VarnameOrigin]:
        """Like :meth:`get`, but also returns the scope object the name was found in.

        Returns a (:class:`pyanalyze.value.Value`, :class:`Scope`, origin) tuple. The :class:`Scope`
        is ``None`` if the name was not found.

        """
        return self.scopes[-1].get(varname, node, state)

    def get_nonlocal_scope(
        self, varname: Varname, using_scope: Scope
    ) -> Optional[Scope]:
        """Gets the defining scope of a non-local variable."""
        for scope in reversed(self.scopes):
            if scope.scope_type is not ScopeType.function_scope:
                continue
            if scope is using_scope:
                continue
            if varname in scope:
                return scope
        return None

    def set(
        self, varname: Varname, value: Value, node: Node, state: VisitorState
    ) -> None:
        """Records an assignment to this variable.

        value is the :term:`value` that is being assigned to `varname`. The other
        arguments are the same as those of :meth:`get`.

        """
        self.scopes[-1].set(varname, value, node, state)

    def suppressing_subscope(self) -> ContextManager[SubScope]:
        return self.scopes[-1].suppressing_subscope()

    def subscope(self) -> ContextManager[SubScope]:
        """Creates a new subscope (see the :class:`FunctionScope` docstring)."""
        return self.scopes[-1].subscope()

    def loop_scope(self) -> ContextManager[List[SubScope]]:
        """Creates a new loop scope (see the :class:`FunctionScope` docstring)."""
        return self.scopes[-1].loop_scope()

    def combine_subscopes(
        self, scopes: Iterable[SubScope], *, ignore_leaves_scope: bool = False
    ) -> None:
        """Merges a number of subscopes back into their parent scope."""
        self.scopes[-1].combine_subscopes(
            scopes, ignore_leaves_scope=ignore_leaves_scope
        )

    def scope_type(self) -> ScopeType:
        """Returns the type of the current scope."""
        return self.scopes[-1].scope_type

    def current_scope(self) -> Scope:
        """Returns the current scope dictionary."""
        return self.scopes[-1]

    def module_scope(self) -> Scope:
        """Returns the module scope of the current scope."""
        return self.scopes[1]

    def contains_scope_of_type(self, scope_type: ScopeType) -> bool:
        """Returns whether any scope in the stack is of this type."""
        return any(scope.scope_type == scope_type for scope in self.scopes)

    def is_nested_function(self) -> bool:
        """Returns whether we're currently in a nested function."""
        return (
            len(self.scopes) > 1
            and self.scopes[-1].scope_type == ScopeType.function_scope
            and self.scopes[-2].scope_type == ScopeType.function_scope
        )


def constrain_value(
    value: Value,
    constraint: AbstractConstraint,
    *,
    simplification_limit: Optional[int] = None,
) -> Value:
    """Create a version of this :term:`value` with the :term:`constraint` applied."""
    return _constrain_value(
        [value], constraint.apply(), simplification_limit=simplification_limit
    )


def uniq_chain(iterables: Iterable[Iterable[T]]) -> List[T]:
    """Returns a flattened list, collapsing equal elements but preserving order."""
    return list(OrderedDict.fromkeys(chain.from_iterable(iterables)))


def _constrain_value(
    values: Sequence[Value],
    constraints: Iterable[Constraint],
    *,
    fallback_value: Optional[Value] = None,
    simplification_limit: Optional[int] = None,
) -> Value:
    # Flatten MultiValuedValue so that we can apply constraints.
    if not values and fallback_value is not None:
        values = list(flatten_values(fallback_value))
    else:
        values = [val for val_or_mvv in values for val in flatten_values(val_or_mvv)]
    for constraint in constraints:
        values = list(constraint.apply_to_values(values))
    if not values:
        return NO_RETURN_VALUE
    if simplification_limit is not None:
        return unite_and_simplify(*values, limit=simplification_limit)
    return unite_values(*values)


def annotate_with_constraint(value: Value, constraint: AbstractConstraint) -> Value:
    if constraint is NULL_CONSTRAINT:
        return value
    return annotate_value(value, [ConstraintExtension(constraint)])


def extract_constraints(value: Value) -> AbstractConstraint:
    if isinstance(value, AnnotatedValue):
        extensions = list(value.get_metadata_of_type(ConstraintExtension))
        constraints = [ext.constraint for ext in extensions]
        base = extract_constraints(value.value)
        constraints = [
            cons for cons in [*constraints, base] if cons is not NULL_CONSTRAINT
        ]
        return AndConstraint.make(constraints)
    elif isinstance(value, MultiValuedValue):
        constraints = [extract_constraints(subval) for subval in value.vals]
        if not constraints:
            return NULL_CONSTRAINT
        return OrConstraint.make(constraints)
    return NULL_CONSTRAINT
