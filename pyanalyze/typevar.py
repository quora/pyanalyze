"""

TypeVar solver.

"""
from typing import Iterable, Sequence, Tuple, Union

import qcore

from .safe import all_of_type
from .value import (
    AnySource,
    AnyValue,
    Bound,
    BoundsMap,
    CanAssignContext,
    CanAssignError,
    IsOneOf,
    LowerBound,
    OrBound,
    TypeVarLike,
    TypeVarMap,
    unite_values,
    UpperBound,
    Value,
)


BOTTOM = qcore.MarkerObject("<bottom>")
TOP = qcore.MarkerObject("<top>")


def resolve_bounds_map(
    bounds_map: BoundsMap,
    ctx: CanAssignContext,
    *,
    all_typevars: Iterable[TypeVarLike] = (),
) -> Tuple[TypeVarMap, Sequence[CanAssignError]]:
    tv_map = {tv: AnyValue(AnySource.generic_argument) for tv in all_typevars}
    errors = []
    for tv, bounds in bounds_map.items():
        bounds = tuple(dict.fromkeys(bounds))
        solution = solve(bounds, ctx)
        if isinstance(solution, CanAssignError):
            errors.append(solution)
            solution = AnyValue(AnySource.error)
        tv_map[tv] = solution
    return tv_map, errors


def solve(
    bounds: Iterable[Bound], ctx: CanAssignContext
) -> Union[Value, CanAssignError]:
    bottom = BOTTOM
    top = TOP
    options = None

    for bound in bounds:
        if isinstance(bound, LowerBound):
            # Ignore lower bounds to Any
            if isinstance(bound.value, AnyValue) and bottom is not BOTTOM:
                continue
            if bottom is BOTTOM or bound.value.is_assignable(bottom, ctx):
                # New bound is more specific. Adopt it.
                bottom = bound.value
            elif bottom.is_assignable(bound.value, ctx):
                # New bound is less specific. Ignore it.
                pass
            else:
                # New bound is separate. We have to satisfy both.
                bottom = unite_values(bottom, bound.value)
        elif isinstance(bound, UpperBound):
            if top is TOP or top.is_assignable(bound.value, ctx):
                top = bound.value
            elif bound.value.is_assignable(top, ctx):
                pass
            else:
                top = unite_values(top, bound.value)
        elif isinstance(bound, OrBound):
            # TODO figure out how to handle this
            continue
        elif isinstance(bound, IsOneOf):
            options = bound.constraints
        else:
            assert False, f"unrecognized bound {bound}"

    if bottom is BOTTOM:
        if top is TOP:
            solution = AnyValue(AnySource.generic_argument)
        else:
            solution = top
    elif top is TOP:
        solution = bottom
    else:
        can_assign = top.can_assign(bottom, ctx)
        if isinstance(can_assign, CanAssignError):
            return CanAssignError(
                "Incompatible bounds on type variable",
                [
                    can_assign,
                    CanAssignError(
                        children=[CanAssignError(str(bound)) for bound in bounds]
                    ),
                ],
            )
        solution = bottom

    if options is not None:
        can_assigns = [option.can_assign(solution, ctx) for option in options]
        if all_of_type(can_assigns, CanAssignError):
            return CanAssignError(children=list(can_assigns))
        available = [
            option
            for option, can_assign in zip(options, can_assigns)
            if not isinstance(can_assign, CanAssignError)
        ]
        # If there's only one solution, pick it.
        if len(available) == 1:
            return available[0]
        # If we inferred Any, keep it; all the solutions will be valid, and
        # picking one will lead to weird errors down the line.
        if isinstance(solution, AnyValue):
            return solution
        available = remove_redundant_solutions(available, ctx)
        if len(available) == 1:
            return available[0]
        # If there are still multiple options, we fall back to Any.
        return AnyValue(AnySource.inference)
    return solution


def remove_redundant_solutions(
    solutions: Sequence[Value], ctx: CanAssignContext
) -> Sequence[Value]:
    # This is going to be quadratic, so don't do it when there's too many
    # opttions.
    initial_count = len(solutions)
    if initial_count > 10:
        return solutions

    temp_solutions = list(solutions)
    for i in range(initial_count):
        sol = temp_solutions[i]
        for j, other in enumerate(temp_solutions):
            if i == j or other is None:
                continue
            if sol.is_assignable(other, ctx) and not other.is_assignable(sol, ctx):
                temp_solutions[i] = None
    return [sol for sol in temp_solutions if sol is not None]
