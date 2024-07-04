"""

Expose an interface for a runtime type checker.

"""

from functools import lru_cache
from typing import Optional

from typing_extensions import deprecated

import pyanalyze

from .annotations import type_from_runtime
from .find_unused import used
from .value import CanAssignError, KnownValue


@lru_cache(maxsize=None)
def _get_checker() -> "pyanalyze.checker.Checker":
    return pyanalyze.checker.Checker()


def is_assignable(value: object, typ: object) -> bool:
    """Return whether ``value`` is assignable to ``typ``.

    This is essentially a more powerful version of ``isinstance()``.
    Examples::

        >>> is_assignable(42, list[int])
        False
        >>> is_assignable([], list[int])
        True
        >>> is_assignable(["x"], list[int])
        False

    The term "assignable" is defined in the typing specification:

        https://typing.readthedocs.io/en/latest/spec/glossary.html#term-assignable

    """
    val = type_from_runtime(typ)
    can_assign = val.can_assign(KnownValue(value), _get_checker())
    return not isinstance(can_assign, CanAssignError)


@used
def get_assignability_error(value: object, typ: object) -> Optional[str]:
    """Return an error message explaining why ``value`` is not
    assignable to ``type``, or None if it is assignable.

    Examples::

        >>> print(get_assignability_error(42, list[int]))
        Cannot assign Literal[42] to list

        >>> print(get_assignability_error([], list[int]))
        None
        >>> print(get_assignability_error(["x"], list[int]))
        In element 0
          Cannot assign Literal['x'] to int

    """
    val = type_from_runtime(typ)
    can_assign = val.can_assign(KnownValue(value), _get_checker())
    if isinstance(can_assign, CanAssignError):
        return can_assign.display(depth=0)
    return None


@used
@deprecated("Use is_assignable instead")
def is_compatible(value: object, typ: object) -> bool:
    """Deprecated alias for is_assignable(). Use that instead."""
    return is_assignable(value, typ)


@used
@deprecated("Use get_assignability_error instead")
def get_compatibility_error(value: object, typ: object) -> Optional[str]:
    """Deprecated alias for get_assignability_error(). Use that instead."""
    return get_assignability_error(value, typ)
