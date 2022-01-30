"""

TypeVar solver.

"""
from typing import Iterable, Sequence, Tuple, Union

from .safe import all_of_type
from .value import (
    NO_RETURN_VALUE,
    AnySource,
    AnyValue,
    Bound,
    BoundsMap,
    CanAssignContext,
    IsOneOf,
    LowerBound,
    OrBound,
    TypeVarLike,
    TypeVarMap,
    UpperBound,
    Value,
    CanAssignError,
    unite_values,
)


BOTTOM = AnyValue(AnySource.generic_argument)
TOP = AnyValue(AnySource.generic_argument)


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
            if bound.value.is_assignable(bottom, ctx):
                # New bound is more specific. Adopt it.
                bottom = bound.value
            elif bottom.is_assignable(bound.value, ctx):
                # New bound is less specific. Ignore it.
                pass
            else:
                # New bound is separate. We have to satisfy both.
                bottom = unite_values(bottom, bound.value)
        elif isinstance(bound, UpperBound):
            if top.is_assignable(bound.value, ctx):
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
        if len(available) == 1:
            return available[0]
        # TODO consider returning unite_values(*available) here instead.
        return AnyValue(AnySource.inference)
    return solution
