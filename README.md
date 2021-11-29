# pyanalyze

Pyanalyze is a tool for programmatically detecting common mistakes in Python code, such as references to undefined variables and type errors.
It can be extended to add additional rules and perform checks specific to particular functions.

Some use cases for this tool include:

- **Catching bugs before they reach production**. The script will catch accidental mistakes like writing "`collections.defalutdict`" instead of "`collections.defaultdict`", so that they won't cause errors in production. Other categories of bugs it can find include variables that may be undefined at runtime, duplicate keys in dict literals, and missing `await` keywords.
- **Making refactoring easier**. When you make a change like removing an object attribute or moving a class from one file to another, pyanalyze will often be able to flag code that you forgot to change.
- **Finding dead code**. It has an option for finding Python objects (functions and classes) that are not used anywhere in the codebase.
- **Checking type annotations**. Type annotations are useful as documentation for readers of code, but only when they are actually correct. Although pyanalyze does not support the full Python type system (see [below](#type-system) for details), it can often detect incorrect type annotations.

## Usage

You can install pyanalyze with:

```bash
$ pip install pyanalyze
```

Once it is installed, you can run pyanalyze on a Python file or package as follows:

```bash
$ python -m pyanalyze file.py
$ python -m pyanalyze package/
```

But note that this will try to import all Python files it is passed. If you have scripts that perform operations without `if __name__ == "__main__":` blocks, pyanalyze may end up executing them.

In order to run successfully, pyanalyze needs to be able to import the code it checks. To make this work you may have to manually adjust Python's import path using the `$PYTHONPATH` environment variable.

Pyanalyze has a number of command-line options, which you can see by running `python -m pyanalyze --help`. Important ones include `-f`, which runs an interactive prompt that lets you examine and fix each error found by pyanalyze, and `--enable`/`--disable`, which enable and disable specific error codes.

### Advanced usage

At Quora, when we want pyanalyze to check a library in CI, we write a unit test that invokes pyanalyze for us. This allows us to run pyanalyze with other tests without further special setup, and it provides a convenient place to put configuration options. An example is pyanalyze's own `test_self.py` test:

```python
import os.path
import pyanalyze
from pyanalyze.error_code import ErrorCode
from pyanalyze.test_node_visitor import skip_before


class PyanalyzeConfig(pyanalyze.config.Config):
    DEFAULT_DIRS = (str(os.path.dirname(__file__)),)
    DEFAULT_BASE_MODULE = pyanalyze
    ENABLED_ERRORS = {
        ErrorCode.possibly_undefined_name,
        ErrorCode.use_fstrings,
        ErrorCode.missing_return_annotation,
        ErrorCode.missing_parameter_annotation,
        ErrorCode.unused_variable,
    }


class PyanalyzeVisitor(pyanalyze.name_check_visitor.NameCheckVisitor):
    config = PyanalyzeConfig()
    should_check_environ_for_files = False


@skip_before((3, 6))
def test_all():
    PyanalyzeVisitor.check_all_files()


if __name__ == "__main__":
    PyanalyzeVisitor.main()
```

### Extending pyanalyze

The main way to extend pyanalyze is by providing a specification for a particular function. This allows you to run arbitrary code that inspects the arguments to the function and raises errors if something is wrong.

As an example, suppose your codebase contains a function `database.run_query()` that takes as an argument a SQL string, like this:

```python
database.run_query("SELECT answer, question FROM content")
```

You want to detect when a call to `run_query()` contains syntactically invalid SQL or refers to a non-existent table or column. You could set that up with code like this:

```python
import pyanalyze
from pyanalyze.error_code import ErrorCode
from pyanalyze.signature import CallContext, Signature, SigParameter
from pyanalyze.value import KnownValue, TypedValue, AnyValue, AnySource, Value

from database import run_query, parse_sql


def run_query_impl(ctx: CallContext) -> Value:
    sql = ctx.vars["sql"]
    if not isinstance(sql, KnownValue) or not isinstance(sql.val, str):
        ctx.show_error(
            "Argument to run_query() must be a string literal",
            ErrorCode.incompatible_call,
        )
        return AnyValue(AnySource.error)

    try:
        parsed = parse_sql(sql)
    except ValueError as e:
        ctx.show_error(
            f"Invalid sql passed to run_query(): {e}",
            ErrorCode.incompatible_call,
        )
        return AnyValue(AnySource.error)

    # check that the parsed SQL is valid...

    # pyanalyze will use this as the inferred return type for the function
    return TypedValue(list)


class Config(pyanalyze.config.Config):
    def get_known_argspecs(self, arg_spec_cache):
        return {
            # This infers the parameter types and names from the function signature
            run_query: arg_spec_cache.get_argspec(
                run_query, impl=run_query_impl
            ),
            # You can also write the signature manually
            run_query: Signature.make(
                [SigParameter("sql", annotation=TypedValue(str))],
                callable=run_query,
                impl=run_query_impl,
            ),
        }
```

### Displaying and checking the type of an expression

You can use `pyanalyze.extensions.reveal_type(expr)` to display the type pyanalyze infers for an expression. This can be
useful to understand errors or to debug why pyanalyze does not catch a particular issue. For example:

```python
from pyanalyze.extensions import reveal_type

reveal_type(1)  # Revealed type is 'Literal[1]' (code: inference_failure)
```

This function is also considered a builtin while type checking, so you can use `reveal_type()` in code that is type checked but not run.

For callable objects, `reveal_type()` will also display the signature inferred by pyanalyze:

```python
from pyanalyze.extensions import reveal_type

reveal_type(reveal_type)  # Revealed type is 'Literal[<function reveal_type at 0x104bf55e0>]', signature is (value, /) -> None (code: inference_failure)
```

A similar function, `pyanalyze.dump_value`, can be used to get lower-level details of the `Value` object pyanalyze infers for an expression.

Similarly, you can use `pyanalyze.assert_is_value` to assert that pyanalyze infers a particular type for
an expression. This requires importing the appropriate `Value` subclass from `pyanalyze.value`. For example:

```python
from pyanalyze import assert_is_value
from pyanalyze.value import KnownValue

assert_is_value(1, KnownValue(1))  # succeeds
assert_is_value(int("2"), KnownValue(1))  # Bad value inference: expected KnownValue(val=1), got TypedValue(typ=<class 'int'>) (code: inference_failure)
```

This function is mostly useful when writing unit tests for pyanalyze or an extension.

### Ignoring errors

Sometimes pyanalyze gets things wrong and you need to ignore an error it emits. This can be done as follows:

- Add `# static analysis: ignore` on a line by itself before the line that generates the erorr.
- Add `# static analysis: ignore` at the end of the line that generates the error.
- Add `# static analysis: ignore` at the top of the file; this will ignore errors in the entire file.

You can add an error code, like `# static analysis: ignore[undefined_name]`, to ignore only a specific error code. This does not work for whole-file ignores. If the `bare_ignore` error code is turned on, pyanalyze will emit an error if you don't specify an error code on an ignore comment.

### Python version support

Pyanalyze supports Python 3.6 through 3.9. Because it imports the code it checks, you have to run it using the same version of Python you use to run your code.

## Type system

Pyanalyze supports most of the Python type system, as specified in [PEP 484](https://www.python.org/dev/peps/pep-0484/) and various later PEPs and in the [Python documentation](https://docs.python.org/3/library/typing.html). It uses type annotations to infer types and checks for type compatibility in calls and return types. Supported type system features include generics like `List[int]`, `NewType`, `TypedDict`, `TypeVar`, and `Callable`.

However, support for some features is still missing or incomplete, including:

- Overloaded functions
- Bounds, constraints, and variance of TypeVars
- `NewType` over non-trivial types
- Missing and required keys in `TypedDict`
- Protocols (PEP 544)
- `ParamSpec` (PEP 612)

### Extensions

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

#### Extended literals

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

#### AsynqCallable

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

#### ParameterTypeGuard

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

#### HasAttrGuard

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

This is also exposed publicly as `pyanalyze.extensions.LiteralOnly`.

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


## Limitations

Python is sufficiently dynamic that almost any check like the ones run by pyanalyze will inevitably have false positives: cases where the script sees an error, but the code in fact runs fine. Attributes may be added at runtime in hard-to-detect ways, variables may be created by direct manipulation of the `globals()` dictionary, and the `mock` module can change anything into anything. Although pyanalyze has a number of whitelists to deal with these false positives, it is usually better to write code in a way that doesn't require use of the whitelist: code that's easier for the script to understand is probably also easier for humans to understand.

Just as the tool inevitably has false positives, it equally inevitably cannot find all code that will throw a runtime error. It is generally impossible to statically determine what a program does or whether it runs successfully without actually running the program. Pyanalyze doesn't check program logic and it cannot always determine exactly what value a variable will have. It is no substitute for unit tests.

## Developing pyanalyze

Pyanalyze has hundreds of unit tests that check its behavior. To run them, you can just run `pytest` in the project directory.

The code is formatted using [Black](https://github.com/psf/black).

## Changelog

Unreleased

- Type check calls with `*args` or `**kwargs` (#275)
- Infer more precise types for comprehensions over known iterables (#279)
- Add impl function for `list.__iadd__` (`+=`) (#280)
- Simplify some overly complex types to improve performance (#280)
- Detect usage of implicitly reexported names (#271)
- Improve type inference for iterables (#277)
- Fix bug in type narrowing for `in`/`not in` (#277)
- Changes affecting consumers of `Value` objects:
  - All `Value` objects are now expected to be hashable.
  - `DictIncompleteValue` and `AnnotatedValue` use tuples instead of lists internally.

Version 0.4.0 (November 18, 2021)

- Support and test Python 3.10. Note that new features are not necessarily
  supported.
- Support PEP 655 (`typing_extensions.Required` and `NotRequired`)
- Improve detection of missing return statements
- Improve detection of suspicious boolean conditions
- The return type of calls with `*args` or `**kwargs` is now inferred
  correctly. The arguments are still not typechecked.
- Fix bug affecting type compatibility between literals and generics
- Improve type narrowing on the `in`/`not in` operator
- Improve type checking for format strings
- Add the `pyanalyze.value.AnyValue` class, replacing `pyanalyze.value.UNRESOLVED_VALUE`
- Improve formatting for `Union` types in errors
- Fix bug affecting type compatibility between types and literals
- Support `total=False` in `TypedDict`
- Deal with typeshed changes in `typeshed_client` 1.1.2
- Better type checking for `list` and `tuple.__getitem__`
- Improve type narrowing on the `==`/`!=` operator
- Reduce usage of `VariableNameValue`
- Improve `TypeVar` inference procedure
- Add support for constraints on the type of `self`, including if it has a union type
- Detect undefined `enum.Enum` members
- Improve handling of `Annotated`
- Add `pyanalyze.extensions.CustomCheck`
- Add `pyanalyze.extensions.ExternalType`
- If you have code dealing with `Value` objects, note that there are several changes:
  - The `UnresolvedValue` class was renamed to `AnyValue`. 
  - `value is UNRESOLVED_VALUE` will no longer be reliable. Use `isinstance(value, AnyValue)` instead.
  - `TypedDictValue` now stores whether each key is required or not in its `items` dictionary.
  - `UnboundMethodValue` now stores a `Composite` object instead of a `Value` object, and has a new
    `typevars` field.
  - There is a new `KnownValueWithTypeVars` class, but it should not be relevant to most use cases.

Version 0.3.1 (August 11, 2021)

- Exit with a non-zero exit code when errors occur
  (contributed by C.A.M. Gerlach)
- Type check the working directory if no command-line arguments
  are given (contributed by C.A.M. Gerlach)

Version 0.3.0 (August 1, 2021)

- Type check calls on Unions properly
- Add `pyanalyze` executable
- Add `--enable-all` and `--disable-all` flags
  (contributed by C.A.M. Gerlach)
- Bug fixes

Version 0.2.0 (May 17, 2021)

- Drop support for Python 2 and 3.5
- Improve unused object finder
- Add support for `TypeVar`
- Add support for `Callable`
- Add `pyanalyze.extensions`
- Add `pyanalyze.ast_annotator`
- Numerous other bug fixes and improvements

Version 0.1.0 (May 29, 2020)

- Initial public release
