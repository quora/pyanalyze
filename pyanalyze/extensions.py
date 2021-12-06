"""

Extensions to the type system supported by pyanalyze. These can be imported at runtime and used in user code.

Several type system extensions are used with the ``Annotated`` type from
`PEP 593 <https://www.python.org/dev/peps/pep-0593/>`_. This allows them to
be gracefully ignored by other type checkers.

"""
from collections import defaultdict
from dataclasses import dataclass, field
import pyanalyze
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    Tuple,
    List,
    Union,
    TypeVar,
    TYPE_CHECKING,
    overload as real_overload,
)
from typing_extensions import Literal, NoReturn

if TYPE_CHECKING:
    from .value import Value, CanAssign, CanAssignContext, TypeVarMap, AnySource


class CustomCheck:
    """A mechanism for extending the type system with user-defined checks.

    To use this, create a subclass of ``CustomCheck`` that overrides the
    ``can_assign`` method, and place it in an ``Annotated`` annotation. The
    return value is equivalent to that of :meth:`pyanalyze.value.Value.can_assign`.

    A simple example is :class:`LiteralOnly`, which is also exposed by pyanalyze
    itself::

        class LiteralOnly(CustomCheck):
            def can_assign(self, value: "Value", ctx: "CanAssignContext") -> "CanAssign":
                for subval in pyanalyze.value.flatten_values(value):
                    if not isinstance(subval, pyanalyze.value.KnownValue):
                        return pyanalyze.value.CanAssignError("Value must be a literal")
                return {}

        def func(arg: Annotated[str, LiteralOnly()]) -> None:
            ...

        func("x")  # ok
        func(str(some_call()))  # error

    A ``CustomCheck`` can also be generic over a ``TypeVar``. To implement support
    for ``TypeVar``, two more methods must be overridden:

    - ``walk_values()`` should yield all ``TypeVar`` objects contained in the check,
      wrapped in a :class:`pyanalyze.value.TypeVarValue`.
    - ``substitute_typevars()`` takes a map from ``TypeVar`` to
      :class:`pyanalyze.value.Value` objects and returns a new ``CustomCheck``.

    """

    def can_assign(self, value: "Value", ctx: "CanAssignContext") -> "CanAssign":
        return {}

    def walk_values(self) -> Iterable["Value"]:
        return []

    def substitute_typevars(self, typevars: "TypeVarMap") -> "CustomCheck":
        return self


@dataclass(frozen=True)
class LiteralOnly(CustomCheck):
    """Custom check that allows only values pyanalyze infers as literals.

    Example:

        def func(arg: Annotated[str, LiteralOnly()]) -> None:
            ...

        func("x")  # ok
        func(str(some_call()))  # error

    This can be useful to prevent user-controlled input in security-sensitive
    APIs.

    """

    def can_assign(self, value: "Value", ctx: "CanAssignContext") -> "CanAssign":
        for subval in pyanalyze.value.flatten_values(value):
            if not isinstance(subval, pyanalyze.value.KnownValue):
                return pyanalyze.value.CanAssignError("Value must be a literal")
        return {}


@dataclass(frozen=True)
class NoAny(CustomCheck):
    """Custom check that disallows passing `Any`."""

    deep: bool = False
    """If true, disallow `Any` in nested positions (e.g., `list[Any]`)."""
    allowed_sources: Container["AnySource"] = field(
        default_factory=lambda: frozenset({pyanalyze.value.AnySource.unreachable})
    )
    """Allow `Any` with these sources."""

    def can_assign(self, value: "Value", ctx: "CanAssignContext") -> "CanAssign":
        if self.deep:
            vals = value.walk_values()
        else:
            vals = pyanalyze.value.flatten_values(value)
        for subval in vals:
            if self._is_disallowed(subval):
                return pyanalyze.value.CanAssignError(f"Value may not be {subval}")
        return {}

    def _is_disallowed(self, value: "Value") -> bool:
        return (
            isinstance(value, pyanalyze.value.AnyValue)
            and value.source not in self.allowed_sources
        )


class _AsynqCallableMeta(type):
    def __getitem__(
        self, params: Tuple[Union[Literal[Ellipsis], List[object]], object]
    ) -> Any:
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError(
                "AsynqCallable[...] should be instantiated "
                "with two arguments (the argument list and a type)."
            )
        if not isinstance(params[0], list) and params[0] is not Ellipsis:
            raise TypeError("The first argument to AsynqCallable must be a list or ...")
        return AsynqCallable(
            Ellipsis if params[0] is Ellipsis else tuple(params[0]), params[1]
        )


@dataclass(frozen=True)
class AsynqCallable(metaclass=_AsynqCallableMeta):
    """Represents an `asynq <https://github.com/quora/asynq>`_ function (a function decorated with ``@asynq()``).

    Similar to ``Callable``, but ``AsynqCallable`` also supports calls
    through ``.asynq()``. Because asynq functions can also be called synchronously,
    an asynq function is assignable to a non-asynq function, but not the reverse.

    The first argument should be the argument list, as for ``Callable``. Examples::

        AsynqCallable[..., int]  # may take any arguments, returns an int
        AsynqCallable[[int], str]  # takes an int, returns a str

    """

    args: Union[Literal[Ellipsis], Tuple[object, ...]]
    return_type: object

    # Returns AsynqCallable but pyanalyze interprets that as AsynqCallable[..., Any]
    def __getitem__(self, item: object) -> Any:
        if not isinstance(item, tuple):
            item = (item,)
        params = self.__parameters__
        if len(params) != len(item):
            raise TypeError(f"incorrect argument count for {self}")
        substitution = dict(zip(params, item))

        def replace_type(arg: object) -> object:
            if isinstance(arg, TypeVar):
                return substitution[arg]
            elif hasattr(arg, "__parameters__"):
                # static analysis: ignore[unsupported_operation]
                return arg[tuple(substitution[param] for param in arg.__parameters__)]
            else:
                return arg

        if self.args is Ellipsis:
            new_args = Ellipsis
        else:
            new_args = tuple(replace_type(arg) for arg in self.args)
        new_return_type = replace_type(self.return_type)
        return AsynqCallable(new_args, new_return_type)

    @property
    def __parameters__(self) -> Tuple["TypeVar", ...]:
        params = []
        for arg in self._inner_types:
            if isinstance(arg, TypeVar):
                params.append(arg)
            elif hasattr(arg, "__parameters__"):
                params += arg.__parameters__
        return tuple(dict.fromkeys(params))

    @property
    def _inner_types(self) -> Iterable[object]:
        if self.args is not Ellipsis:
            yield from self.args
        yield self.return_type

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"{self} is not callable")


class _ParameterTypeGuardMeta(type):
    def __getitem__(self, params: Tuple[str, object]) -> "ParameterTypeGuard":
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError(
                "ParameterTypeGuard[...] should be instantiated "
                "with two arguments (a variable name and a type)."
            )
        if not isinstance(params[0], str):
            raise TypeError("The first argument to ParameterTypeGuard must be a string")
        return ParameterTypeGuard(params[0], params[1])


@dataclass(frozen=True)
class ParameterTypeGuard(metaclass=_ParameterTypeGuardMeta):
    """A guard on an arbitrary parameter. Used with ``Annotated``.

    Example usage::

        def is_int(arg: object) -> Annotated[bool, ParameterTypeGuard["arg", int]]:
            return isinstance(arg, int)

    """

    varname: str
    guarded_type: object


class _HasAttrGuardMeta(type):
    def __getitem__(self, params: Tuple[str, str, object]) -> "HasAttrGuard":
        if not isinstance(params, tuple) or len(params) != 3:
            raise TypeError(
                "HasAttrGuard[...] should be instantiated "
                "with three arguments (a variable name, an attribute name, and a type)."
            )
        if not isinstance(params[0], str):
            raise TypeError("The first argument to HasAttrGuard must be a string")
        return HasAttrGuard(params[0], params[1], params[2])


@dataclass(frozen=True)
class HasAttrGuard(metaclass=_HasAttrGuardMeta):
    """A guard on an arbitrary parameter that checks for the presence of an attribute.
    Used with ``Annotated``.

    A return type of ``Annotated[bool, HasAttrGuard[param, attr, type]]`` means that
    `param` has an attribute named `attr` of type `type` if the function
    returns True.

    Example usage::

        def has_time(arg: object) -> Annotated[bool, HasAttrGuard["arg", Literal["time"], int]]:
            attr = getattr(arg, "time", None)
            return isinstance(attr, int)

        T = TypeVar("T", bound=str)

        def hasattr(obj: object, name: T) -> Annotated[bool, HasAttrGuard["obj", T, Any]]:
            try:
                getattr(obj, name)
                return True
            except AttributeError:
                return False

    """

    varname: str
    attribute_name: object
    attribute_type: object


class _TypeGuardMeta(type):
    def __getitem__(self, params: object) -> "TypeGuard":
        return TypeGuard(params)


@dataclass(frozen=True)
class TypeGuard(metaclass=_TypeGuardMeta):
    """Type guards, as defined in `PEP 647 <https://www.python.org/dev/peps/pep-0647/>`_.

    New code should instead use ``typing_extensions.TypeGuard`` or
    (in Python 3.10 and higher) ``typing.TypeGuard``.

    Example usage::

        def is_int_list(arg: list[Any]) -> TypeGuard[list[int]]:
            return all(isinstance(elt, int) for elt in arg)

    """

    guarded_type: object


class _ExternalTypeMeta(type):
    def __getitem__(self, params: str) -> "ExternalType":
        if not isinstance(params, str):
            raise TypeError(f"ExternalType expects a string, not {params!r}")
        return ExternalType(params)


@dataclass(frozen=True)
class ExternalType(metaclass=_ExternalTypeMeta):
    """`ExternalType` is a way to refer to a type that is not imported at runtime.
    The type must be given as a string representing a fully qualified name.

    Example usage::

        from pyanalyze.extensions import ExternalType

        def function(arg: ExternalType["other_module.Type"]) -> None:
            pass

    To resolve the type, pyanalyze will import `other_module`, but the module
    using `ExternalType` does not have to import `other_module`.

    `typing.TYPE_CHECKING` can be used in a similar fashion, but `ExternalType`
    can be more convenient when programmatically generating types. Our motivating
    use case is our database schema definition file: we would like to map each
    column to the enum it corresponds to, but those enums are defined in code
    that should not be imported by the schema definition.

    """

    type_path: str

    # This makes it possible to use ExternalType within e.g. Annotated
    def __call__(self) -> NoReturn:
        raise NotImplementedError("just here to fool typing._type_check")


def reveal_type(value: object) -> None:
    """Inspect the inferred type of an expression.

    Calling this function will make pyanalyze print out the argument's
    inferred value in a human-readable format. At runtime it does nothing.

    This is automatically exposed as a global during type checking, so in
    code that is not run at import, `reveal_type()` can be used without
    being impoorted.

    Example::

        def f(x: int) -> None:
            reveal_type(x)  # Revealed type is "int"

    """
    pass


_overloads: Dict[str, List[Callable[..., Any]]] = defaultdict(list)


def get_overloads(fully_qualified_name: str) -> List[Callable[..., Any]]:
    """Return all defined runtime overloads for this fully qualified name."""
    return _overloads[fully_qualified_name]


if TYPE_CHECKING:
    from typing import overload as overload

else:

    def overload(func: Callable[..., Any]) -> Callable[..., Any]:
        """A version of `typing.overload` that is inspectable at runtime.

        If this decorator is used for a function `some_module.some_function`, calling
        :func:`pyanalyze.extensiions.get_overloads("some_module.some_function")` will
        return all the runtime overloads.

        """
        key = f"{func.__module__}.{func.__qualname__}"
        _overloads[key].append(func)
        return real_overload(func)
