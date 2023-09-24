"""

Expose an interface for a runtime type checker.

"""

from typing import Optional
import pyanalyze
from functools import lru_cache

from .value import CanAssignError, KnownValue
from .annotations import type_from_runtime
from .find_unused import used


@lru_cache(maxsize=None)
def _get_checker() -> "pyanalyze.checker.Checker":
    return pyanalyze.checker.Checker()


@used
def is_compatible(value: object, typ: object) -> bool:
    """Return whether ``value`` is compatible with ``type``.

    Examples::

        >>> is_compatible(42, list[int])
        False
        >>> is_compatible([], list[int])
        True
        >>> is_compatible(["x"], list[int])
        False

    """
    val = type_from_runtime(typ)
    can_assign = val.can_assign(KnownValue(value), _get_checker())
    return not isinstance(can_assign, CanAssignError)


@used
def get_compatibility_error(value: object, typ: object) -> Optional[str]:
    """Return an error message explaining why ``value`` is not
    compatible with ``type``, or None if they are compatible.

    Examples::

        >>> print(get_compatibility_error(42, list[int]))
        Cannot assign Literal[42] to list

        >>> print(get_compatibility_error([], list[int]))
        None
        >>> print(get_compatibility_error(["x"], list[int]))
        In element 0
          Cannot assign Literal['x'] to int

    """
    val = type_from_runtime(typ)
    can_assign = val.can_assign(KnownValue(value), _get_checker())
    if isinstance(can_assign, CanAssignError):
        return can_assign.display(depth=0)
    return None
