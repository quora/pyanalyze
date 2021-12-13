"""

Implementation of scope nesting in pyanalyze.

This module is responsible for mapping names to their values in pyanalyze. Variable lookup happens
mostly through a series of nested dictionaries. When pyanalyze sees a reference to a name inside a
nested function, it will first look at that function's scope, then in the enclosing function's
scope, then in the module scope, and finally in the builtin scope containing Python builtins. Each
of these scopes is represented as a :class:`Scope` object, which by default is just a thin wrapper around a
dictionary. However, function scopes are more complicated in order to track variable values
accurately through control flow structures like if blocks. See the :class:`FunctionScope` docstring for
details.

Other subtleties implemented here:

- Multiple assignments to the same name result in :class:`pyanalyze.value.MultiValuedValue`
- Globals are represented as :class:`pyanalyze.value.ReferencingValue`, and name lookups for such names are delegated to
  the :class:`pyanalyze.value.ReferencingValue`\'s scope
- Class scopes except the current one are skipped in name lookup

"""
from ast import AST
from collections import defaultdict, OrderedDict
import contextlib
from dataclasses import dataclass, field, replace
import enum
import qcore
from itertools import chain
import builtins
from types import ModuleType
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Sequence,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .boolability import get_boolability
from .extensions import reveal_type
from .safe import safe_equals, safe_issubclass
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    KnownValue,
    ReferencingValue,
    SubclassValue,
    TypeVarMap,
    TypedValue,
    Value,
    annotate_value,
    UNINITIALIZED_VALUE,
    unite_and_simplify,
    unite_values,
    flatten_values,
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
    attributes: Sequence[Union[str, KnownValue]]

    def extend_with(self, index: Union[str, KnownValue]) -> "CompositeVariable":
        return CompositeVariable(self.varname, (*self.attributes, index))


Varname = Union[str, CompositeVariable]
# Nodes as used in scopes can be any object, as long as they are hashable.
Node = object
SubScope = Dict[Varname, List[Node]]

# Type for Constraint.value if constraint type is predicate
# PredicateFunc = Callable[[Value, bool], Optional[Value]]


class Composite(NamedTuple):
    """A :class:`pyanalyze.value.Value` with information about its
    origin. This is useful for setting constraints."""

    value: Value
    varname: Optional[Varname] = None
    node: Optional[AST] = None

    def get_extended_varname(
        self, index: Union[str, KnownValue]
    ) -> Optional[CompositeVariable]:
        if self.varname is None:
            return None
        if isinstance(self.varname, str):
            return CompositeVariable(self.varname, (index,))
        else:
            return self.varname.extend_with(index)

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
    :class:`AndConstraint` and :class:`OrConstraint`, but unlike these, all constraints in a `one_of`
    or `all_of` constraint apply to the same :term:`varname`.
    """
    all_of = 5
    """All of several other constraints on `varname` are true."""
    is_value_object = 6
    """`constraint.varname` should be typed as a :class:`pyanalyze.value.Value` object. Naming of this
    and `is_value` is confusing, and ideally we'd come up with better names."""
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
        """Yields concrete constraints that are active when this constraint is applied."""
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

    varname: Varname
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

    def apply(self) -> Iterable["Constraint"]:
        yield self

    def invert(self) -> "Constraint":
        return Constraint(
            self.varname, self.constraint_type, not self.positive, self.value
        )

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
                    # _constrain_values() will eventually return AnyValue.
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
                for val in constraint.apply_to_value(value):
                    yield val

        elif self.constraint_type == ConstraintType.all_of:
            vals = [value]
            for constraint in self.value:
                vals = list(constraint.apply_to_values(vals))
            for applied in vals:
                yield applied

        else:
            assert False, f"unknown constraint type {self.constraint_type}"


TRUTHY_CONSTRAINT = Constraint("%unused", ConstraintType.is_truthy, True, None)
FALSY_CONSTRAINT = Constraint("%unused", ConstraintType.is_truthy, False, None)


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

    varname: Varname
    provider: Callable[[Value], Value]

    def apply(self) -> Iterable[Constraint]:
        return []

    def invert(self) -> AbstractConstraint:
        # inverting is meaningless
        return NULL_CONSTRAINT


@dataclass(frozen=True)
class AndConstraint(AbstractConstraint):
    """Represents the AND of two constraints."""

    left: AbstractConstraint
    right: AbstractConstraint

    def apply(self) -> Iterable["Constraint"]:
        for constraint in self.left.apply():
            yield constraint
        for constraint in self.right.apply():
            yield constraint

    def invert(self) -> "OrConstraint":
        # ~(A and B) -> ~A or ~B
        return OrConstraint(self.left.invert(), self.right.invert())


@dataclass(frozen=True)
class OrConstraint(AbstractConstraint):
    """Represents the OR of two constraints."""

    left: AbstractConstraint
    right: AbstractConstraint

    def apply(self) -> Iterable[Constraint]:
        left = self._group_constraints(self.left)
        right = self._group_constraints(self.right)
        for varname, constraints in left.items():
            # Produce one_of constraints if the same variable name
            # applies on both the left and the right side.
            if varname in right:
                yield Constraint(
                    varname,
                    ConstraintType.one_of,
                    True,
                    [
                        self._constraint_from_list(varname, constraints),
                        self._constraint_from_list(varname, right[varname]),
                    ],
                )

    def _constraint_from_list(
        self, varname: Varname, constraints: Sequence[Constraint]
    ) -> Constraint:
        if len(constraints) == 1:
            return constraints[0]
        else:
            return Constraint(varname, ConstraintType.all_of, True, constraints)

    def _group_constraints(
        self, abstract_constraint: AbstractConstraint
    ) -> Dict[str, List[Constraint]]:
        by_varname = defaultdict(list)
        for constraint in abstract_constraint.apply():
            by_varname[constraint.varname].append(constraint)
        return by_varname

    def invert(self) -> AndConstraint:
        # ~(A or B) -> ~A and ~B
        return AndConstraint(self.left.invert(), self.right.invert())


class _ConstrainedValue(Value):
    """Helper class, only used within a FunctionScope."""

    def __init__(
        self, definition_nodes: Set[Node], constraints: Sequence[Constraint]
    ) -> None:
        self.definition_nodes = definition_nodes
        self.constraints = constraints
        self.resolution_cache = {}


_empty_constrained = _ConstrainedValue(set(), [])


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
    ) -> Tuple[Value, Optional["Scope"]]:
        local_value = self.get_local(
            varname, node, state, from_parent_scope=from_parent_scope
        )
        if local_value is not UNINITIALIZED_VALUE:
            return self.resolve_reference(local_value, state), self
        elif self.parent_scope is not None:
            # Parent scopes don't get the node to help local lookup.
            parent_node = (
                (varname, self.scope_node) if self.scope_node is not None else None
            )
            return self.parent_scope.get(
                varname, parent_node, state, from_parent_scope=True
            )
        else:
            return UNINITIALIZED_VALUE, None

    def get_local(
        self,
        varname: Varname,
        node: Node,
        state: VisitorState,
        from_parent_scope: bool = False,
        fallback_value: Optional[Value] = None,
    ) -> Value:
        if varname in self.variables:
            return self.variables[varname]
        else:
            return UNINITIALIZED_VALUE

    def set(
        self, varname: Varname, value: Value, node: Node, state: VisitorState
    ) -> None:
        if varname not in self:
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

    def items(self) -> Iterable[Tuple[Varname, Value]]:
        return self.variables.items()

    def __contains__(self, varname: Varname) -> bool:
        return varname in self.variables

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
            referenced, _ = value.scope.get(value.name, None, state)
            # globals that are None are probably set to something else later
            if safe_equals(referenced, KnownValue(None)):
                return AnyValue(AnySource.inference)
            else:
                return referenced
        else:
            return value

    def scope_used_as_parent(self) -> "Scope":
        """Class scopes are skipped in scope lookup, so don't set them as parent scopes."""
        if self.scope_type == ScopeType.class_scope:
            assert (
                self.parent_scope is not None
            ), "class scopes must have a parent scope"
            return self.parent_scope.scope_used_as_parent()
        else:
            return self


class FunctionScope(Scope):
    """Keeps track of the local variables of a single function.

    :class:`FunctionScope` is designed to produce the correct value for each variable at each point in the
    function, unlike the base :class:`Scope` class, which assumes that each variable has the same value
    throughout the scope it represents.

    For example, given the code::

        x = 3
        x = 4
        print(x)

    :class:`FunctionScope` will infer the value of `x` to be ``KnownValue(4)``, but :class:`Scope` will produce a
    :class:`pyanalyze.value.MultiValuedValue` because it does not know whether the assignment to 3 or 4 is active.

    The approach taken is to map each usage node (a place where the variable is used) to a set of
    definition nodes (places where the variable is assigned to) that could be active when the
    variable is used. Each definition node is also mapped to the value assigned to the variable
    there.

    For example, in the code::

        x = 3  # (a)
        print(x)  # (b)

    (a) is the only definition node for the usage node at (b), and (a) is mapped to ``KnownValue(3)``,
    so at (b) x is inferred to be ``KnownValue(3)``.

    However, in this code::

        if some_condition():
            x = 3  # (a)
        else:
            x = 4  # (b)
        print(x)  # (c)

    both (a) and (b) are possible definition nodes for the usage node at (c), so at (c) x is
    inferred to be a ``MultiValuedValue([KnownValue(3), KnownValue(4)])``.

    These mappings are implemented as the `usage_to_definition_nodes` and `definition_node_to_value`
    attributes on the :class:`FunctionScope` object. They are created completely during the collecting
    :term:`phase`. The basic mechanism uses the `name_to_current_definition_nodes` dictionary, which maps
    each local variable to a list of active definition nodes. When pyanalyze encounters an
    assignment, it updates `name_to_current_definition_nodes` to map to that assignment node, and
    when it encounters a variable usage it updates `usage_to_definition_nodes` to map that usage
    to the current definition nodes in `name_to_current_definition_nodes`. For example::

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
    handle this case, :class:`FunctionScope` supports the creation of subscopes and the `combine_subscopes`
    operation. Each branch in a conditional statement is mapped to a separate subscope, which
    contains an independently updated copy of `name_to_current_definition_nodes`. After pyanalyze
    visits all branches, it runs the `combine_subscopes` operation on all of the branches' subscopes.
    This operation takes, for each variable, the union of the definition nodes created in all of the
    branches. For example::

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

    a naive approach would infer that `x` is ``None`` at (a). To take care of this case, pyanalyze visits
    the loop body twice during the collecting :term:`phase`, so that `usage_to_definition_nodes` can add a
    mapping of (a) to (b). To handle `break` and `continue` correctly, it also uses a separate "loop
    scope" that ends up combining the scopes created by normal control flow through the body of the
    loop and by each `break` and `continue` statement.

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
    usage_to_definition_nodes: Dict[Node, List[Node]]
    definition_node_to_value: Dict[Node, Value]
    name_to_all_definition_nodes: Dict[str, Set[Node]]
    name_to_composites: Dict[str, Set[CompositeVariable]]
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
            def_nodes = set(self.name_to_current_definition_nodes[constraint.varname])
            # We set both a constraint and its inverse using the same node as the definition
            # node, so cheat and include the constraint itself in the key. If you write constraints
            # to the same key in definition_node_to_value multiple times, you're likely to get
            # infinite recursion.
            node = (node, constraint)
            assert (
                node not in self.definition_node_to_value
            ), "duplicate constraint for {}".format(node)
            self.definition_node_to_value[node] = _ConstrainedValue(
                def_nodes, [constraint]
            )
            self.name_to_current_definition_nodes[constraint.varname] = [node]
            self._add_composite(constraint.varname)

    def set(
        self, varname: Varname, value: Value, node: Node, state: VisitorState
    ) -> None:
        if isinstance(value, ReferencingValue):
            self.referencing_value_vars[varname] = value
            return
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

    def get_local(
        self,
        varname: str,
        node: Node,
        state: VisitorState,
        from_parent_scope: bool = False,
        fallback_value: Optional[Value] = None,
    ) -> Value:
        self._add_composite(varname)
        ctx = _LookupContext(varname, fallback_value, node, state)
        if from_parent_scope:
            self.accessed_from_special_nodes.add(varname)
        if node is None:
            self.accessed_from_special_nodes.add(varname)
            # this indicates that we're not looking at a normal local variable reference, but
            # something special like a nested function
            if varname in self.name_to_all_definition_nodes:
                return self._get_value_from_nodes(
                    self.name_to_all_definition_nodes[varname], ctx
                )
            else:
                return self.referencing_value_vars[varname]
        if state is VisitorState.check_names:
            if node not in self.usage_to_definition_nodes:
                return self.referencing_value_vars[varname]
            else:
                definers = self.usage_to_definition_nodes[node]
        else:
            if varname in self.name_to_current_definition_nodes:
                definers = self.name_to_current_definition_nodes[varname]
                self.usage_to_definition_nodes[node] += definers
            else:
                return self.referencing_value_vars[varname]
        return self._get_value_from_nodes(definers, ctx)

    @contextlib.contextmanager
    def subscope(self) -> Iterable[SubScope]:
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
    def loop_scope(self) -> Iterable[SubScope]:
        loop_scopes = []
        with self.subscope() as main_scope:
            loop_scopes.append(main_scope)
            with qcore.override(self, "current_loop_scopes", loop_scopes):
                yield main_scope
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
            if val.definition_nodes or ctx.fallback_value:
                resolved = self._get_value_from_nodes(
                    val.definition_nodes, ctx, val.constraints
                )
            else:
                assert (
                    self.parent_scope
                ), "constrained value must have definition nodes or parent scope"
                parent_val, _ = self.parent_scope.get(ctx.varname, None, ctx.state)
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

    def __contains__(self, varname: Varname) -> bool:
        return varname in self.name_to_all_definition_nodes


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
    ) -> Iterable[None]:
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
    def ignore_topmost_scope(self) -> Iterable[None]:
        """Context manager that temporarily ignores the topmost scope."""
        scope = self.scopes.pop()
        try:
            yield
        finally:
            self.scopes.append(scope)

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

        Returns :data:`pyanalyze.value.UNINITIALIZED_VALUE` if the name is not defined in any known scope.

        """
        value, _ = self.get_with_scope(varname, node, state)
        return value

    def get_with_scope(
        self, varname: Varname, node: Node, state: VisitorState
    ) -> Tuple[Value, Optional[Scope]]:
        """Like :meth:`get`, but also returns the scope object the name was found in.

        Returns a (:class:`pyanalyze.value.Value`, :class:`Scope`) tuple. The :class:`Scope` is ``None`` if the name was not found.

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

    def subscope(self) -> ContextManager[SubScope]:
        """Creates a new subscope (see the :class:`FunctionScope` docstring)."""
        return self.scopes[-1].subscope()

    def loop_scope(self) -> ContextManager[None]:
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
    values: Iterable[Value],
    constraints: Iterable[Constraint],
    *,
    fallback_value: Optional[Value] = None,
    simplification_limit: Optional[int] = None,
) -> Value:
    # Flatten MultiValuedValue so that we can apply constraints.
    values = [val for val_or_mvv in values for val in flatten_values(val_or_mvv)]
    if not values and fallback_value is not None:
        values = list(flatten_values(fallback_value))
    for constraint in constraints:
        values = list(constraint.apply_to_values(values))
    if not values:
        # TODO: maybe show an error here? This branch should mean the code is
        # unreachable.
        return AnyValue(AnySource.unreachable)
    if simplification_limit is not None:
        return unite_and_simplify(*values, limit=simplification_limit)
    return unite_values(*values)
