"""

"Safe" operations that call into user code and catch any excxeptions.

"""
from typing import Any, Tuple, Union, Container, Type, TypeVar, Iterable
from typing_extensions import Annotated

from .extensions import ParameterTypeGuard

T = TypeVar("T")


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


def safe_issubclass(cls: type, class_or_tuple: Union[type, Tuple[type, ...]]) -> bool:
    try:
        return issubclass(cls, class_or_tuple)
    except Exception:
        return False


def safe_isinstance(obj: object, class_or_tuple: Union[type, Tuple[type, ...]]) -> bool:
    try:
        return isinstance(obj, class_or_tuple)
    except Exception:
        return False


def safe_in(item: T, collection: Container[T]) -> bool:
    """Safely check whether item is in collection. Defaults to returning false."""
    # Workaround against mock objects sometimes throwing ValueError if you compare them,
    # and against objects throwing other kinds of errors if you use in.
    try:
        return item in collection
    except Exception:
        return False


def is_hashable(obj: object) -> bool:
    try:
        hash(obj)
    except Exception:
        return False
    else:
        return True


def is_iterable(obj: object) -> bool:
    """Returns whether a Python object is iterable."""
    typ = type(obj)
    if hasattr(typ, "__iter__"):
        return True
    return hasattr(typ, "__getitem__") and hasattr(typ, "__len__")


def all_of_type(
    elts: Iterable[object], typ: Type[T]
) -> Annotated[bool, ParameterTypeGuard["elts", Iterable[T]]]:
    """Returns whether all elements of elts are instances of typ."""
    return all(isinstance(elt, typ) for elt in elts)
