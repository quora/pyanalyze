"""

Extensions to the type system supported by pyanalyze.

"""
from dataclasses import dataclass
from typing import Tuple, TypeVar

_T = TypeVar("_T")


class _ParameterTypeGuardMeta(type):
    def __getitem__(self, params: Tuple[str, object]) -> "ParameterTypeGuard":
        if not isinstance(params, tuple) or len(params) < 2:
            raise TypeError(
                "ParameterTypeGuard[...] should be instantiated "
                "with two arguments (a variable name and a type)."
            )
        if not isinstance(params[0], str):
            raise TypeError("The first argument to ParameterTypeGuard must be a string")
        return ParameterTypeGuard(params[0], params[1])


@dataclass
class ParameterTypeGuard(metaclass=_ParameterTypeGuardMeta):
    """A guard on an arbitrary parameter.

    Example usage:

        def is_int(arg: object) -> Annotated[bool, ParameterTypeGuard["arg", int]]:
            return isinstance(arg, int)

    """

    varname: str
    guarded_type: object
