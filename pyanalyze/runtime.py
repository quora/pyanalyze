"""

Expose an interface for a runtime type checker.

"""

from typing import Optional
from pyanalyze.value import CanAssignError, KnownValue
from .annotations import type_from_runtime
from .checker import Checker
from .find_unused import used

_CHECKER = Checker()


@used
def is_compatible(typ: object, value: object) -> bool:
    """Return whether ``value`` is compatible with ``type``.

    Examples::

        >>> is_compatible(list[int], 42)
        False
        >>> is_compatible(list[int], [])
        True
        >>> is_compatible(list[int], ["x"])
        False

    """
    val = type_from_runtime(typ)
    can_assign = val.can_assign(KnownValue(value), _CHECKER)
    return not isinstance(can_assign, CanAssignError)


@used
def get_compatibility_error(typ: object, value: object) -> Optional[str]:
    """Return an error message explaining why ``value`` is not
    compatible with ``type``, or None if they are compatible.

    Examples::

        >>> print(get_compatibility_error(list[int], 42))
        Cannot assign Literal[42] to list

        >>> print(get_compatibility_error(list[int], []))
        None
        >>> print(get_compatibility_error(list[int], ["x"]))
        In element 0
          Cannot assign Literal['x'] to int

    """
    val = type_from_runtime(typ)
    can_assign = val.can_assign(KnownValue(value), _CHECKER)
    if isinstance(can_assign, CanAssignError):
        return can_assign.display(depth=0)
    return None
