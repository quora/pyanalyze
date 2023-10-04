# Type system

Pyanalyze supports most of the Python type system, as specified in [PEP 484](https://www.python.org/dev/peps/pep-0484/) and various later PEPs and in the [Python documentation](https://docs.python.org/3/library/typing.html). It uses type annotations to infer types and checks for type compatibility in calls and return types. Supported type system features include generics like `List[int]`, `NewType`, `TypedDict`, `TypeVar`, and `Callable`.

## Extensions

In addition to the standard Python type system, pyanalyze supports a number of non-standard extensions:

- Callable literals: you can declare a parameter as `Literal[some_function]` and it will accept any callable assignable to `some_function`. Pyanalyze also supports Literals of various other types in addition to those supported by [PEP 586](https://www.python.org/dev/peps/pep-0586/).
- `pyanalyze.extensions.AsynqCallable` is a variant of `Callable` that applies to `asynq` functions.
- `pyanalyze.extensions.ParameterTypeGuard` is a generalization of PEP 649's `TypeGuard` that allows guards on any parameter to a function. To use it, return `Annotated[bool, ParameterTypeGuard["arg", SomeType]]`.
- `pyanalyze.extensions.HasAttrGuard` is a similar mechanism that allows indicating that an object has a particular attribute. To use it, return `Annotated[bool, HasAttrGuard["arg", "attribute", SomeType]]`.
- `pyanalyze.extensions.ExternalType` is a way to refer to a type that cannot
  be referenced by name in contexts where using `if TYPE_CHECKING` is not possible.
- `pyanalyze.extensions.CustomCheck` is a powerful mechanism to extend the type system
  with custom user-defined checks.

They are explained in more detail below.

### Extended literals

Literal types are specified by [PEP 586](https://www.python.org/dev/peps/pep-0586/). The PEP only supports Literals of int, str, bytes, bool, Enum, and None objects, but pyanalyze accepts Literals over all Python objects.

As an extension, pyanalyze accepts any compatible callable for a Literal over a function type. This allows more flexible callable types.

For example:

```python
from typing_extensions import Literal

def template(x: int, y: str = "") -> None:
    pass

def takes_template(func: Literal[template]) -> None:
    func(x=1, y="x")

def good_callable(x: int, y: str = "default", z: float = 0.0) -> None:
    pass

takes_template(good_callable)  # accepted

def bad_callable(not_x: int, y: str = "") -> None:
    pass

takes_template(bad_callable)  # rejected
```

### AsynqCallable

The `@asynq()` callable in the [asynq](https://www.github.com/quora/asynq) framework produces a special callable that can either be called directly (producing a synchronous call) or through the special `.asynq()` attribute (producing an asynchronous call). The `AsynqCallable` special form is similar to `Callable`, but describes a callable with this extra `.asynq()` attribute.

For example, this construct can be used to implement the `asynq.tools.amap` helper function:

```python
from asynq import asynq
from pyanalyze.extensions import AsynqCallable
from typing import TypeVar, List, Iterable

T = TypeVar("T")
U = TypeVar("U")

@asynq()
def amap(function: AsynqCallable[[T], U], sequence: Iterable[T]) -> List[U]:
    return (yield [function.asynq(elt) for elt in sequence])
```

Because of limitations in the runtime typing library, some generic aliases involving AsynqCallable will not work at runtime. For example, given a generic alias `L = List[AsynqCallable[[T], int]]`, `L[str]` will throw an error. Quoting the type annotation works around this.

### ParameterTypeGuard

[PEP 647](https://www.python.org/dev/peps/pep-0647/) added support for type guards, a mechanism to narrow the type of a variable. However, it only supports narrowing the first argument to a function.

Pyanalyze supports an extended version that combines with [PEP 593](https://www.python.org/dev/peps/pep-0593/)'s `Annotated` type to support guards on any function parameter.

For example, the below function narrows the type of two of its parameters:

```python
from typing import Iterable, Annotated
from pyanalyze.extensions import ParameterTypeGuard
from pyanalyze.value import KnownValue, Value


def _can_perform_call(
    args: Iterable[Value], keywords: Iterable[Value]
) -> Annotated[
    bool,
    ParameterTypeGuard["args", Iterable[KnownValue]],
    ParameterTypeGuard["keywords", Iterable[KnownValue]],
]:
    return all(isinstance(arg, KnownValue) for arg in args) and all(
        isinstance(kwarg, KnownValue) for kwarg in keywords
    )
```

### HasAttrGuard

`HasAttrGuard` is similar to `ParameterTypeGuard` and `TypeGuard`, but instead of narrowing a type, it indicates that an object has a particular attribute. For example, consider this function:

```python
from typing import Literal, Annotated
from pyanalyze.extensions import HasAttrGuard

def has_time(arg: object) -> Annotated[bool, HasAttrGuard["arg", Literal["time"], int]]:
    attr = getattr(arg, "time", None)
    return isinstance(attr, int)
```

After a call to `has_time(o)` succeeds, pyanalyze will know that `o.time` exists and is of type `int`.

In practice the main use of this type is to implement the type of `hasattr` itself. In pure Python `hasattr` could look like this:

```python
from typing import Any, TypeVar, Annotated
from pyanalyze.extensions import HasAttrGuard

T = TypeVar("T", bound=str)

def hasattr(obj: object, name: T) -> Annotated[bool, HasAttrGuard["obj", T, Any]]:
    try:
        getattr(obj, name)
        return True
    except AttributeError:
        return False
```

As currently implemented, `HasAttrGuard` does not narrow types; instead it preserves the previous type of a variable and adds the additional attribute.

### ExternalType

`ExternalType` is a way to refer to a type that is not imported at runtime.
The type must be fully qualified.

```python
from pyanalyze.extensions import ExternalType

def function(arg: ExternalType["other_module.Type"]) -> None:
    pass
```

To resolve the type, pyanalyze will import `other_module`, but the module
using `ExternalType` does not have to import `other_module`.

`typing.TYPE_CHECKING` can be used in a similar fashion, but `ExternalType`
can be more convenient when programmatically generating types. Our motivating
use case is our database schema definition file: we would like to map each
column to the enum it corresponds to, but those enums are defined in code
that should not be imported by the schema definition.

### CustomCheck

`CustomCheck` is a mechanism that allows users to define additional checks
that are not natively supported by the type system. To use it, create a
new subclass of `CustomCheck` that overrides the `can_assign` method. Such
objects can then be placed in `Annotated` annotations.

For example, the following creates a custom check that allows only literal
values:

```python
from pyanalyze.extensions import CustomCheck
from pyanalyze.value import Value, CanAssign, CanAssignContext, CanAssignError, KnownValue, flatten_values

class LiteralOnly(CustomCheck):
    def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        for subval in flatten_values(value):
            if not isinstance(subval, KnownValue):
                return CanAssignError("Value must be a literal")
        return {}
```

It is used as follows:

```python
def func(arg: Annotated[str, LiteralOnly()]) -> None:
    ...

func("x")  # ok
func(str(some_call()))  # error
```

Custom checks can also be generic. For example, the following custom check
implements basic support for integers with a limited range:

```python
from dataclasses import dataclass
from pyanalyze.extensions import CustomCheck
from pyanalyze.value import (
    AnyValue,
    flatten_values,
    CanAssign,
    CanAssignError,
    CanAssignContext,
    KnownValue,
    TypeVarMap,
    TypeVarValue,
    Value,
)
from typing_extensions import Annotated, TypeGuard
from typing import Iterable, TypeVar, Union

# Annotated[] annotations must be hashable
@dataclass(frozen=True)
class GreaterThan(CustomCheck):
    # The value can be either an integer or a TypeVar. In the latter case,
    # the check hasn't been specified yet, and we let everything through.
    value: Union[int, TypeVar]

    def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(self.value, TypeVar):
            return {}
        # flatten_values() unwraps unions, but we don't want to unwrap
        # Annotated, so we can accept other Annotated objects.
        for subval in flatten_values(value, unwrap_annotated=False):
            if isinstance(subval, AnnotatedValue):
                # If the inner value isn't valid, error immediately (for example,
                # if it's an int that's too small).
                can_assign = self._can_assign_inner(subval.value)
                if not isinstance(can_assign, CanAssignError):
                    return can_assign
                gts = list(subval.get_custom_check_of_type(GreaterThan))
                if not gts:
                    # We reject values that are just ints with no GreaterThan
                    # annotation.
                    return CanAssignError(f"Size of {value} is not known")
                # If a value winds up with multiple GreaterThan annotations,
                # we allow it if at least one is bigger than or equal to our value.
                if not any(
                    check.value >= self.value
                    for check in gts
                    if isinstance(check.value, int)
                ):
                    return CanAssignError(f"{subval} is too small")
            else:
                can_assign = self._can_assign_inner(subval)
                if isinstance(can_assign, CanAssignError):
                    return can_assign
        return {}

    def _can_assign_inner(self, value: Value) -> CanAssign:
        if isinstance(value, KnownValue):
            if not isinstance(value.val, int):
                return CanAssignError(f"Value {value.val!r} is not an int")
            if value.val <= self.value:
                return CanAssignError(
                    f"Value {value.val!r} is not greater than {self.value}"
                )
        elif isinstance(value, AnyValue):
            # We let Any through.
            return {}
        else:
            # Should be mostly TypedValue.
            return CanAssignError(f"Size of {value} is not known")

    def walk_values(self) -> Iterable[Value]:
        if isinstance(self.value, TypeVar):
            yield TypeVarValue(self.value)

    def substitute_typevars(self, typevars: TypeVarMap) -> "GreaterThan":
        if isinstance(self.value, TypeVar) and self.value in typevars:
            value = typevars[self.value]
            if isinstance(value, KnownValue) and isinstance(value.val, int):
                return GreaterThan(value.val)
        return self

def more_than_two(x: Annotated[int, GreaterThan(2)]) -> None:
    pass

IntT = TypeVar("IntT", bound=int)

def is_greater_than(
    x: int, limit: IntT
) -> TypeGuard[Annotated[int, GreaterThan(IntT)]]:
    return x > limit

def caller(x: int) -> None:
    more_than_two(x)  # E: incompatible_argument
    if is_greater_than(x, 2):
        more_than_two(x)  # ok
    more_than_two(3)  # ok
    more_than_two(2)  # E: incompatible_argument
```

This is not a full, usable implementation of ranged integers; for that we would
also need to add support for this check to operators like `int.__add__`.

Two custom checks are exposed by `pyanalyze.extensions`:

- `pyanalyze.extensions.LiteralOnly`, which allows only literal values (as discussed above)
- `pyanalyze.extensions.NoAny`, which disallows passing untyped values

## Limitations

Although pyanalyze aims to support the full Python type system, support for some features is still missing or incomplete, including:

- Variance of TypeVars
- `NewType` over non-trivial types
- `ParamSpec` (PEP 612)
- `TypeVarTuple` (PEP 646)

More generally, Python is sufficiently dynamic that almost any check like the ones run by pyanalyze will inevitably have false positives: cases where the script sees an error, but the code in fact runs fine. Attributes may be added at runtime in hard-to-detect ways, variables may be created by direct manipulation of the `globals()` dictionary, and the `unittest.mock` module can change anything into anything. Although pyanalyze has a number of configuration mechanisms to deal with these false positives, it is usually better to write code in a way that doesn't require use of these knobs: code that's easier for the script to understand is probably also easier for humans to understand.

Just as the tool inevitably has false positives, it equally inevitably cannot find all code that will throw a runtime error. It is generally impossible to statically determine what a program does or whether it runs successfully without actually running the program. Pyanalyze doesn't check program logic and it cannot always determine exactly what value a variable will have. It is no substitute for unit tests.
