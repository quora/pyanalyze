"""

"Safe" operations that call into user code and catch any excxeptions.

"""
from typing import Any, Tuple, Union


def safe_hasattr(item: object, member: str) -> bool:
    try:
        # some sketchy implementation (like paste.registry) of
        # __getattr__ caused errors at static analysis.
        return hasattr(item, member)
    except Exception:
        return False


def safe_getattr(value: object, attr: str, default: object) -> Any:
    """Returns whether this value has the given attribute, ignoring exceptions."""
    try:
        return getattr(value, attr)
    except Exception:
        return default


def safe_equals(left: object, right: object) -> bool:
    try:
        return bool(left == right)
    except Exception:
        return False


def safe_issubclass(value: type, typ: Union[type, Tuple[type, ...]]) -> bool:
    try:
        return issubclass(value, typ)
    except Exception:
        return False


def is_hashable(obj: object) -> bool:
    try:
        hash(obj)
    except Exception:
        return False
    else:
        return True
