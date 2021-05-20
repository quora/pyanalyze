"""

Extensions to the type system supported by pyanalyze. These can be imported at runtime and used in user code.

Several type system extensions are used with the ``Annotated`` type from
`PEP 593 <https://www.python.org/dev/peps/pep-0593/>`_. This allows them to
be gracefully ignored by other type checkers.

"""
from dataclasses import dataclass
from typing import Any, Iterable, Tuple, List, Union, TypeVar
from typing_extensions import Literal


class _AsynqCallableMeta(type):
    def __getitem__(
        self, params: Tuple[Union[Literal[Ellipsis], List[object]], object]
    ) -> "AsynqCallable":
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

    def __getitem__(self, item: object) -> "AsynqCallable":
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
