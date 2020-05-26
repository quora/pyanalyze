from __future__ import absolute_import, print_function, division, unicode_literals

"""

Implementation of scope nesting in pyanalyze.

This module is responsible for mapping names to their values in pyanalyze. Variable lookup happens
mostly through a series of nested dictionaries. When pyanalyze sees a reference to a name inside a
nested function, it will first look at that function's scope, then in the enclosing function's
scope, then in the module scope, and finally in the builtin scope containing Python builtins. Each
of these scopes is represented as a Scope object, which by default is just a thin wrapper around a
dictionary. However, function scopes are more complicated in order to track variable values
accurately through control flow structures like if blocks. See the FunctionScope docstring for
details.

Other subtleties implemented here:
- Multiple assignments to the same name result in MultiValuedValue
- Globals are represented as ReferencingValues, and name lookups for such names are delegated to
  the ReferencingValue's scope
- Class scopes except the current one are skipped in name lookup

"""

from collections import defaultdict, namedtuple, OrderedDict
import contextlib
import enum
import qcore
from qcore.asserts import assert_is_instance
from itertools import chain
import six
from six.moves import builtins

from .value import (
    KnownValue,
    ReferencingValue,
    SubclassValue,
    TypedValue,
    Value,
    boolean_value,
    UNINITIALIZED_VALUE,
    UNRESOLVED_VALUE,
    unite_values,
    flatten_values,
)


LEAVES_SCOPE = "%LEAVES_SCOPE"
LEAVES_LOOP = "%LEAVES_LOOP"
_UNINITIALIZED = qcore.MarkerObject("uninitialized")
_LookupContext = namedtuple(
    "_LookupContext", ["varname", "fallback_value", "node", "state"]
)


class VisitorState(enum.Enum):
    collect_names = 1
    check_names = 2


class ScopeType(enum.Enum):
    builtin_scope = 1
    module_scope = 2
    class_scope = 3
    function_scope = 4


class ConstraintType(enum.Enum):
    # corresponds to (not) isinstance(constraint.varname, constraint.value)
    is_instance = 1
    # corresponds to constraint.varname is (not) constraint.value
    is_value = 2
    # corresponds to if (not) constraint.varname
    is_truthy = 3
    # For these constraint types, the value is itself a list of constraints. These
    # constraints are always positive. They are similar to the abstract
    # AndConstraint and OrConstraint, but unlike these, all constraints in a one_of
    # or all_of constraint apply to the same variable.
    # at least one of several other constraints on varname is true
    one_of = 4
    # all of several other constraints on varname are true
    all_of = 5


class AbstractConstraint(qcore.InspectableClass):
    """Base class for abstract constraints.

    We distinguish between abstract and concrete constraints. Abstract
    constraints are collected from conditions, and may be null constraints,
    concrete constraints, or an AND or OR of other abstract constraints.
    When we add constraints to a scope, we apply the abstract constraints to
    produce a set of concrete constraints. For example, a null constraint
    produces no concrete constraints, and an AND constraint AND(C1, C2)
    produces both C1 and C2.

    Concrete constraints are instances of the Constraint class.

    """

    def apply(self):
        """Yields concrete constraints that are active when this constraint is applied."""
        raise NotImplementedError

    def invert(self):
        """Return an inverted version of this constraint."""
        raise NotImplementedError

    def __hash__(self):
        # Constraints need to be hashable by identity.
        return object.__hash__(self)


class Constraint(AbstractConstraint):
    """A constraint is a restriction on the value of a variable.

    Constraints are tracked in scope objects, so that we know which constraints
    are active for a given usage of a variable.

    Constraints have the following attributes:
    - varname (str): name of the variable the constraint applies to
    - constraint_type (ConstraintType): type of constraint
    - positive (bool): whether this is a positive constraint or not (e.g.,
      for an is_truthy constraint, "if x" would lead to a positive and "if not x"
      to a negative constraint)
    - value: type for an is_instance constraint; value identical to the variable
      for is_value; unused for is_truthy.

    For example:

        def f(x: Optional[int]) -> None:
            # x can be either an int or None
            assert x
            # Now a constraint of type is_truthy is active. Because
            # None is not truthy, we now know that x is of type int.

    """

    def __init__(self, varname, constraint_type, positive, value):
        self.varname = varname
        self.constraint_type = constraint_type
        self.positive = positive
        self.value = value

    def apply(self):
        yield self

    def invert(self):
        """Returns the opposite of this constraint."""
        return Constraint(
            self.varname, self.constraint_type, not self.positive, self.value
        )

    def apply_to_values(self, values):
        for value in values:
            for applied in self.apply_to_value(value):
                yield applied

    def apply_to_value(self, value):
        """Yield values consistent with this constraint.

        Produces zero or more values consistent both with the given
        value and with this constraint.

        The value may not be a MultiValuedValue.

        """
        if value is UNINITIALIZED_VALUE:
            yield UNINITIALIZED_VALUE
            return
        if self.constraint_type == ConstraintType.is_instance:
            if value is UNRESOLVED_VALUE:
                if self.positive:
                    yield TypedValue(self.value)
                else:
                    yield UNRESOLVED_VALUE
            elif isinstance(value, KnownValue):
                if self.positive:
                    if isinstance(value.val, self.value):
                        yield value
                else:
                    if not isinstance(value.val, self.value):
                        yield value
            elif isinstance(value, TypedValue):
                if self.positive:
                    if _safe_issubclass(value.typ, self.value):
                        yield value
                    elif _safe_issubclass(self.value, value.typ):
                        yield TypedValue(self.value)
                    # TODO: Technically here we should infer an intersection type:
                    # a type that is a subclass of both types. In practice currently
                    # _constrain_values() will eventually return UNRESOLVED_VALUE.
                else:
                    if not _safe_issubclass(value.typ, self.value):
                        yield value
            elif isinstance(value, SubclassValue):
                if self.positive:
                    if isinstance(value.typ, self.value):
                        yield value
                else:
                    if not isinstance(value.typ, self.value):
                        yield value

        elif self.constraint_type == ConstraintType.is_value:
            if self.positive:
                known_val = KnownValue(self.value)
                if value is UNRESOLVED_VALUE:
                    yield known_val
                elif isinstance(value, KnownValue):
                    if value.val is self.value:
                        yield known_val
                elif isinstance(value, TypedValue):
                    if isinstance(self.value, value.typ):
                        yield known_val
                elif isinstance(value, SubclassValue):
                    if isinstance(self.value, type) and _safe_issubclass(
                        self.value, value.typ
                    ):
                        yield known_val
            else:
                if not (isinstance(value, KnownValue) and value.val is self.value):
                    yield value

        elif self.constraint_type == ConstraintType.is_truthy:
            if self.positive:
                if boolean_value(value) is not False:
                    yield value
            else:
                if boolean_value(value) is not True:
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
            assert False, "unknown constraint type %s" % (self.constraint_type,)


class NullConstraint(AbstractConstraint):
    """Represents the absence of a constraint."""

    def apply(self):
        return []

    def invert(self):
        return self


NULL_CONSTRAINT = NullConstraint()


class AndConstraint(AbstractConstraint):
    """Represents the AND of two constraints."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def apply(self):
        for constraint in self.left.apply():
            yield constraint
        for constraint in self.right.apply():
            yield constraint

    def invert(self):
        # ~(A and B) -> ~A or ~B
        return OrConstraint(self.left.invert(), self.right.invert())


class OrConstraint(AbstractConstraint):
    """Represents the OR of two constraints."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def apply(self):
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
                        self._constraint_from_list(varname, left[varname]),
                        self._constraint_from_list(varname, right[varname]),
                    ],
                )

    def _constraint_from_list(self, varname, constraints):
        if len(constraints) == 1:
            return constraints[0]
        else:
            return Constraint(varname, ConstraintType.all_of, True, constraints)

    def _group_constraints(self, abstract_constraint):
        by_varname = defaultdict(list)
        for constraint in abstract_constraint.apply():
            by_varname[constraint.varname].append(constraint)
        return by_varname

    def invert(self):
        # ~(A or B) -> ~A and ~B
        return AndConstraint(self.left.invert(), self.right.invert())


CompositeVariable = namedtuple("CompositeVariable", ["varname", "attributes"])

# __doc__ is not writable in Python 2
if six.PY3:
    CompositeVariable.__doc__ = """Fake variable used to implement constraints on instance variables.

For example, access to "self.x" would make us use
CompositeVariable('self', ('x',)). If a function contains a check for
isinstance(self.x, int), we would put a Constraint on this CompositeVariable.

"""


class _ConstrainedValue(Value):
    """Helper class, only used within a FunctionScope."""

    def __init__(self, definition_nodes, constraints):
        self.definition_nodes = definition_nodes
        self.constraints = constraints
        self.resolution_cache = {}


_empty_constrained = _ConstrainedValue(set(), [])


class Scope(qcore.InspectableClass):
    """Represents a single level in the scope stack.

    May be a builtin, module, class, or function scope.

    """

    def __init__(self, scope_type, variables, parent_scope, scope_node=None):
        assert_is_instance(scope_type, ScopeType)
        self.scope_type = scope_type
        assert_is_instance(variables, dict)
        self.variables = variables
        if parent_scope is not None:
            self.parent_scope = parent_scope.scope_used_as_parent()
        else:
            self.parent_scope = None
        self.scope_node = scope_node

    def add_constraint(self, abstract_constraint, node, state):
        """Constraints are ignored outside of function scopes."""
        pass

    def get(self, varname, node, state):
        local_value = self.get_local(varname, node, state)
        if local_value is not UNINITIALIZED_VALUE:
            return self.resolve_reference(local_value, state)
        elif self.parent_scope is not None:
            # Parent scopes don't get the node to help local lookup.
            parent_node = (varname, self.scope_node) if self.scope_node else None
            return self.parent_scope.get(varname, parent_node, state)
        else:
            return UNINITIALIZED_VALUE

    def get_local(self, varname, node, state):
        if varname in self.variables:
            return self.variables[varname]
        else:
            return UNINITIALIZED_VALUE

    def set(self, varname, value, node, state):
        if varname not in self:
            self.variables[varname] = value
        elif value is UNRESOLVED_VALUE or not _safe_equals(
            self.variables[varname], value
        ):
            existing = self.variables[varname]
            if isinstance(existing, ReferencingValue):
                existing.scope.set(existing.name, value, node, state)
            elif (
                type(existing) is TypedValue
                and isinstance(value, TypedValue)
                and existing.typ is value.typ
            ):
                # replace with a more specific TypedValue
                self.variables[varname] = value
            else:
                self.variables[varname] = unite_values(existing, value)

    def items(self):
        return six.iteritems(self.variables)

    def __contains__(self, varname):
        return varname in self.variables

    # no real subscopes in non-function scopes, just dummy implementations
    @contextlib.contextmanager
    def subscope(self):
        yield

    @contextlib.contextmanager
    def loop_scope(self):
        """Context manager for the subscope associated with a loop."""
        yield

    def combine_subscopes(self, scopes):
        pass

    def resolve_reference(self, value, state):
        if isinstance(value, ReferencingValue):
            referenced = value.scope.get(value.name, None, state)
            # globals that are None are probably set to something else later
            if _safe_equals(referenced, KnownValue(None)):
                return UNRESOLVED_VALUE
            else:
                return referenced
        else:
            return value

    def scope_used_as_parent(self):
        """Class scopes are skipped in scope lookup, so don't set them as parent scopes."""
        if self.scope_type == ScopeType.class_scope:
            return self.parent_scope.scope_used_as_parent()
        else:
            return self


class FunctionScope(Scope):
    """Keeps track of the local variables of a single function.

    FunctionScope is designed to produce the correct value for each variable at each point in the
    function, unlike the base Scope class, which assumes that each variable has the same value
    throughout the scope it represents.

    For example, given the code:

        x = 3
        x = 4
        print(x)

    FunctionScope will infer the value of x to be KnownValue(4), but Scope will produce a
    MultiValuedValue because it does not know whether the assignment to 3 or 4 is active.

    The approach taken is to map each usage node (a place where the variable is used) to a set of
    definition nodes (places where the variable is assigned to) that could be active when the
    variable is used. Each definition node is also mapped to the value assigned to the variable
    there.

    For example, in the code:

        x = 3  # (a)
        print(x)  # (b)

    (a) is the only definition node for the usage node at (b), and (a) is mapped to KnownValue(3),
    so at (b) x is inferred to be KnownValue(3).

    However, in this code:

        if some_condition():
            x = 3  # (a)
        else:
            x = 4  # (b)
        print(x)  # (c)

    both (a) and (b) are possible definition nodes for the usage node at (c), so at (c) x is
    inferred to be a MultiValuedValue([KnownValue(3), KnownValue(4)]).

    These mappings are implemented as the usage_to_definition_nodes and definition_node_to_value
    attributes on the FunctionScope object. They are created completely during the collecting
    phase. The basic mechanism uses the name_to_current_definition_nodes dictionary, which maps
    each local variable to a list of active definition nodes. When pyanalyze encounters an
    assignment, it updates name_to_current_definition_nodes to map to that assignment node, and
    when it encounters a variable usage it updates usage_to_definition_nodes to map that usage
    to the current definition nodes in name_to_current_definition_nodes. For example:

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
    handle this case, FunctionScope supports the creation of subscopes and the combine_subscopes
    operation. Each branch in a conditional statement is mapped to a separate subscope, which
    contains an independently updated copy of name_to_current_definition_nodes. After pyanalyze
    visits all branches, it runs the combine_subscopes operation on all of the branches' subscopes.
    This operation takes, for each variable, the union of the definition nodes created in all of the
    branches. For example:

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
    definition nodes later in the same loop body. For example, in code like this:

        x = None
        for _ in (1, 2):
            if x:
                print(x[1])  # (a)
            else:
                x = (1, 2)  # (b)

    a naive approach would infer that x is None at (a). To take care of this case, pyanalyze visits
    the loop body twice during the collecting phase, so that usage_to_definition_nodes can add a
    mapping of (a) to (b). To handle break and continue correctly, it also uses a separate "loop
    scope" that ends up combining the scopes created by normal control flow through the body of the
    loop and by each break and continue statement.

    Try-finally blocks are handled by visiting the finally block twice. Essentially, we treat:

        try:
            TRY-BODY
        finally:
            FINALLY-BODY
        REST-OF-FUNCTION

    as equivalent to:

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
    all. A for loop of the form:

        for TARGET in ITERABLE:
            FOR-BODY
        else:
            ELSE-BODY

    is treated like:

        if <empty>:
            TARGET = next(iter(ITERABLE))
            FOR-BODY
        else:
            ELSE_BODY

    Special logic is also needed to take care of globals (which are kept track of separately from
    normal variables) and variable lookups without a node context. For the latter, the
    name_to_all_definition_nodes maps each variable name to all possible definition nodes.

    """

    def __init__(self, parent_scope, scope_node=None):
        super(FunctionScope, self).__init__(
            ScopeType.function_scope, {}, parent_scope, scope_node
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

    def add_constraint(self, abstract_constraint, node, state):
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

    def set(self, varname, value, node, state):
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

    def get_local(self, varname, node, state, fallback_value=None):
        self._add_composite(varname)
        ctx = _LookupContext(varname, fallback_value, node, state)
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
    def subscope(self):
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
    def loop_scope(self):
        loop_scopes = []
        with self.subscope() as main_scope:
            loop_scopes.append(main_scope)
            with qcore.override(self, "current_loop_scopes", loop_scopes):
                yield
        self.combine_subscopes(
            [
                {name: values for name, values in scope.items() if name != LEAVES_LOOP}
                for scope in loop_scopes
            ]
        )

    def get_combined_scope(self, scopes):
        new_scopes = []
        for scope in scopes:
            if LEAVES_LOOP in scope:
                self.current_loop_scopes.append(scope)
            elif LEAVES_SCOPE not in scope:
                new_scopes.append(scope)
        if not new_scopes:
            return {LEAVES_SCOPE: [UNRESOLVED_VALUE]}
        all_variables = set(chain.from_iterable(new_scopes))
        return {
            varname: _uniq_chain(
                scope.get(varname, [_UNINITIALIZED]) for scope in new_scopes
            )
            for varname in all_variables
        }

    def combine_subscopes(self, scopes):
        self.name_to_current_definition_nodes.update(self.get_combined_scope(scopes))

    def _resolve_value(self, val, ctx):
        if isinstance(val, _ConstrainedValue):
            # Cache repeated resolutions of the same ConstrainedValue, because otherwise
            # lots of nested constraints can lead to exponential performance (see the
            # test_repeated_constraints test case).
            key = ctx._replace(fallback_value=None)
            if key in val.resolution_cache:
                return val.resolution_cache[key]
            if val.definition_nodes or ctx.fallback_value:
                resolved = self._get_value_from_nodes(
                    val.definition_nodes, ctx, val.constraints
                )
            else:
                parent_val = self.parent_scope.get(ctx.varname, None, ctx.state)
                resolved = _constrain_value([parent_val], val.constraints)
            val.resolution_cache[key] = resolved
            return resolved
        else:
            return val

    def _get_value_from_nodes(self, nodes, ctx, constraints=()):
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
        return _constrain_value(values, constraints, fallback_value=ctx.fallback_value)

    def _add_composite(self, varname):
        if isinstance(varname, CompositeVariable):
            self.name_to_composites[varname.varname].add(varname)
            if len(varname.attributes) > 1:
                for i in range(1, len(varname.attributes)):
                    composite = CompositeVariable(
                        varname.varname, varname.attributes[:i]
                    )
                    self.name_to_composites[composite].add(varname)

    def items(self):
        raise NotImplementedError

    def __contains__(self, varname):
        return varname in self.name_to_all_definition_nodes


class StackedScopes(object):
    """Represents the stack of scopes in which Python searches for variables."""

    _builtin_scope = Scope(
        ScopeType.builtin_scope,
        {k: KnownValue(v) for k, v in six.iteritems(builtins.__dict__)},
        None,
    )

    def __init__(self, module):
        if module is None:
            module_vars = {"__name__": TypedValue(str), "__file__": TypedValue(str)}
        else:
            module_vars = {
                key: KnownValue(value) for key, value in six.iteritems(module.__dict__)
            }
        self.scopes = [
            self._builtin_scope,
            Scope(ScopeType.module_scope, module_vars, self._builtin_scope),
        ]

    @contextlib.contextmanager
    def add_scope(self, scope_type, scope_node):
        """Contextmanager that temporarily adds a scope of this type to the top of the stack."""
        if scope_type is ScopeType.function_scope:
            scope = FunctionScope(self.scopes[-1], scope_node)
        else:
            scope = Scope(scope_type, {}, self.scopes[-1], scope_node)
        self.scopes.append(scope)
        try:
            yield
        finally:
            self.scopes.pop()

    @contextlib.contextmanager
    def ignore_topmost_scope(self):
        """Context manager that temporarily ignores the topmost scope."""
        scope = self.scopes.pop()
        try:
            yield
        finally:
            self.scopes.append(scope)

    def get(self, varname, node, state):
        """Gets a variable of the given name from the current scope stack.

        Arguments:
        - varname: name of the variable to retrieve
        - node: AST node corresponding to the place where the variable lookup is happening.
          FunctionScope uses this to decide which definition of the variable to use; other scopes
          ignore it. It can be passed as None to indicate that any definition may be used. This is
          used among others when looking up names in outer scopes. Although this argument should
          normally be an AST node, it can be any unique, hashable identifier, because sometimes a
          single AST node sets multiple variables (e.g. in ImportFrom nodes).
        - state: the current VisitorState. pyanalyze runs the collecting state to collect all name
          assignments and map name usages to their corresponding assignments, and then the checking
          state to locate any errors in the code.

        Raises NameError if the name is not defined in any known scope.

        """
        return self.scopes[-1].get(varname, node, state)

    def get_nonlocal_scope(self, varname, using_scope):
        """Gets the defining scope of a non-local variable."""
        for scope in reversed(self.scopes):
            if scope.scope_type is not ScopeType.function_scope:
                continue
            if scope is using_scope:
                continue
            if varname in scope:
                return scope
        else:
            return None

    def set(self, varname, value, node, state):
        """Records an assignment to this variable.

        value is the value that is being assigned to varname. It should be an instance of
        value.Value. The other arguments are the same as those of get().

        """
        self.scopes[-1].set(varname, value, node, state)

    def subscope(self):
        """Creates a new subscope (see the FunctionScope docstring)."""
        return self.scopes[-1].subscope()

    def loop_scope(self):
        """Creates a new loop scope (see the FunctionScope docstring)."""
        return self.scopes[-1].loop_scope()

    def combine_subscopes(self, scopes):
        """Merges a number of subscopes back into their parent scope."""
        self.scopes[-1].combine_subscopes(scopes)

    def scope_type(self):
        """Returns the type of the current scope."""
        return self.scopes[-1].scope_type

    def current_scope(self):
        """Returns the current scope dictionary."""
        return self.scopes[-1]

    def module_scope(self):
        """Returns the module scope of the current scope."""
        return self.scopes[1]

    def contains_scope_of_type(self, scope_type):
        """Returns whether any scope in the stack is of this type."""
        return any(scope.scope_type == scope_type for scope in self.scopes)

    def is_nested_function(self):
        """Returns whether we're currently in a nested function."""
        return (
            len(self.scopes) > 1
            and self.scopes[-1].scope_type == ScopeType.function_scope
            and self.scopes[-2].scope_type == ScopeType.function_scope
        )


def _uniq_chain(iterables):
    """Returns a flattened list, collapsing equal elements but preserving order."""
    return list(OrderedDict.fromkeys(chain.from_iterable(iterables)))


def _safe_equals(left, right):
    try:
        return bool(left == right)
    except Exception:
        return False


def _safe_issubclass(value, typ):
    try:
        return issubclass(value, typ)
    except Exception:
        return False


def _constrain_value(values, constraints, fallback_value=None):
    # Flatten MultiValuedValue so that we can apply constraints.
    values = [val for val_or_mvv in values for val in flatten_values(val_or_mvv)]
    if not values and fallback_value is not None:
        values = list(flatten_values(fallback_value))
    for constraint in constraints:
        values = list(constraint.apply_to_values(values))
    if not values:
        # TODO: maybe show an error here? This branch should mean the code is
        # unreachable.
        return UNRESOLVED_VALUE
    return unite_values(*values)
