"""

"Safe" operations that call into user code and catch any exceptions.

"""
import inspect
import sys
from typing import Any, Tuple, Union, Container, NewType, Type, TypeVar, Iterable
from typing_extensions import Annotated

from .extensions import ParameterTypeGuard

T = TypeVar("T")


def safe_hasattr(item: object, member: str) -> bool:
    """Safe version of ``hasattr()``."""
    try:
        # some sketchy implementation (like paste.registry) of
        # __getattr__ cause hasattr() to throw an error.
        return hasattr(item, member)
    except Exception:
        return False


def safe_getattr(value: object, attr: str, default: object) -> Any:
    """Whether this value has the given attribute, ignoring exceptions."""
    try:
        return getattr(value, attr)
    except Exception:
        return default


def safe_equals(left: object, right: object) -> bool:
    """Safely check whether two objects are equal."""
    try:
        return bool(left == right)
    except Exception:
        return False


def safe_issubclass(cls: type, class_or_tuple: Union[type, Tuple[type, ...]]) -> bool:
    """Safe version of ``issubclass()``.

    Apart from incorrect arguments, ``issubclass(a, b)`` can throw an error
    only if `b` has a ``__subclasscheck__`` method that throws an error.
    Therefore, it is not necessary to use ``safe_issubclass()`` if the class
    is known to not override ``__subclasscheck__``.

    """
    try:
        return issubclass(cls, class_or_tuple)
    except Exception:
        return False


def safe_isinstance(obj: object, class_or_tuple: Union[type, Tuple[type, ...]]) -> bool:
    """Safe version of ``isinstance()``.

    Apart from incorrect arguments, ``isinstance(a, b)`` can throw an error
    only if `b` has a ``__instancecheck__`` method that throws an error.
    Therefore, it is not necessary to use ``safe_isinstance()`` if the class
    is known to not override ``__instancecheck__``.

    """
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
    """Return whether an object is hashable."""
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


if sys.version_info >= (3, 10):

    def is_newtype(obj: object) -> bool:
        return isinstance(obj, NewType)


else:

    def is_newtype(obj: object) -> bool:
        return (
            inspect.isfunction(obj)
            and hasattr(obj, "__supertype__")
            and isinstance(obj.__supertype__, type)
        )
