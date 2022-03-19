# Type evaluation

Type evaluation is a mechanism for replacing complex
overloads and version checks. It provides a restricted
subset of Python that can be executed by type checkers
to customize the behavior of a particular function.

## Motivation

Consider the
definition of `round()` in typeshed:

    @overload
    def round(number: SupportsRound[Any]) -> int: ...
    @overload
    def round(number: SupportsRound[Any], ndigits: None) -> int: ...
    @overload
    def round(number: SupportsRound[_T], ndigits: SupportsIndex) -> _T: ...

With type evaluation, this could instead be written as:

    @evaluated
    def round(number: SupportsRound[_T], ndigits: SupportsIndex | None = None):
        if ndigits is None:
            return int
        else:
            return _T

This makes it easier to see at a glance what the difference is between various overloads.

Other features of type evaluation, as proposed here, include
customizable error messages, branching on the type of an
argument, and branching on whether an argument was provided
as a positional or keyword argument.

Type evaluation functions can replace most complex overloads
with simpler, more readable code. They solve a number of
problems:

- Type evaluation functions provide ways to implement
  several type system features that have been previously
  requested, including:
  - Marking a function, parameter, or parameter type as
    deprecated.
  - Accepting `Sequence[str]` but not `str`.
  - Checking whether two generic arguments are overlapping
- Error messages involving overloads are often hard to read.
  Type evaluation functions enable the author of a function
  to provide custom error messages that clearly point out
  the issue in the user's code.
- Complex overloads can be difficult to understand and write.
  Type evaluation functions provide a more natural interface
  that is closer to how such functions are written at
  runtime.
- The precise behavior of overloads is not specified and
  varies across type checkers. The behavior of type
  evaluation functions is more precisely specified.

## Specification

This section specifies how type evaluation works, without
commentary. Discussion, with motivating use cases, is
provided in the "Discussion" section below. The examples
in this section are meant to clarify the semantics only.

Type-evaluated functions may be declared both at runtime and in stub files, similar to the existing `@overload` mechanism. At runtime, the evaluated function must immediately precede
the function implementation:

    @evaluated
    def round(number: SupportsRound[_T], ndigits: SupportsIndex | None = None):
        if ...

    def round(number, ndigits=None):
        return number.__round__(ndigits)

In stubs, the implementation is omitted.

When a type checker encounters a call to a function for which a type evaluation has been provided, it should
do the following:

- Validate that the arguments to the call are compatible
  with the type annotations on the parameters to the
  evaluation function, as with a normal call.
- Symbolically evaluate the body of the type evaluation
  until it reaches a `return` statement, which provides the type
  that the call should return. During this symbolic
  evaluation, each argument is set to the value it has at
  the call site that is being evaluated.
- If execution reached a `return` statement, return the type
  provided by that statement. Otherwise, return the type
  set in the evaluation function's return annotation, or
  `Any` if there is no return annotation.

Type checkers are encouraged to provide a strictness option
that produces an error if an evaluation function is missing
a type annotation on a parameter or return type. However, no
error should be provided if the return annotation is missing
and all branches (including error branches) return a type.

The default value of a parameter to an evaluation function
may be either `...` or any value that is valid inside
`Literal[...]`. If an argument with default `X` is not
provided in a call, the type of the argument within the
evaluation function is `Literal[X]`. If the default is
`...`, the type is the parameter's annotation instead.

Simple examples to demonstrate the semantics:

    @evaluated
    def always_returns(x: int):
        return str

    always_returns("x")  # error: "x" is not an int
    always_returns()  # error: not enough arguments
    reveal_type(always_returns(1))  # str

    @evaluated
    def always_errors(x: int):
        show_error("error")

    x = always_errors(1)  # error
    reveal_type(x)  # Any

    @evaluated
    def always_errors_with_type(x: int) -> str:
        show_error("error")

    x = always_errors(1)  # error
    reveal_type(x)  # str

    @evaluated
    def with_defaults(x: int = ..., y: int = 1) -> None:
        reveal_type(x)
        reveal_type(y)

    with_defaults()  # x is "int", y is "Literal[1]"
    with_defaults(1)  # x and y are both "Literal[1]"

### Supported features

The body of a type evaluation uses a restricted subset of Python.
The only supported features are:

- `if` statements and `else` blocks. These can only contain conditions of the form specified below.
- `return` statements with return values that are interpretable as type annotations. This indicates the type that the function returns in a particular condition.
- `pass` statements, which do nothing.
- Calls to `show_error()`, which cause the type checker
  to emit an error. These are discussed further below.
- Calls to `reveal_type(arg)`, where arg is one of the
  arguments to the type evaluation function. These cause the
  type checker to emit a message showing the current type
  of `arg`. This is a debugging feature.

Conditions in `if` statements may contain:

- A call to one of the following functions, which are covered
  in more detail below:
  - `is_provided()`, which returns whether a parameter was
    explicitly provided in a call.
  - `is_positional()`, which returns whether a parameter
    was provided through a positional argument.
  - `is_keyword()`, which returns whether a parameter
    was provided through a keyword argument.
  - `is_of_type()`, which returns whether a parameter is of
    a particular type.
- Expressions of the form `arg <op> <constant>`, where `<op>`
  is one of `is`, `is not`, `==`, or `!=`. This is equivalent
  to `(not) is_of_type(arg, Literal[<constant>], exclude_any=True)`.
  `<constant>` may be any value that is valid inside `Literal` (`None`, a string, a bool, an int, or an enum
  member).
- Version and platform checks that are otherwise valid in stubs, as specified in PEP 484.
- Multiple conditions combined with `and` or `or`.
- A negation of another condition with `not`.

### show_error()

The `show_error()` special function has the following
signature:

    def show_error(message: str, /, *, argument: Any | None = ...): ...

The `message` parameter must be a string literal.
Calls to this function cause the type checker to emit an
error that includes the given message.
Execution continues past the `show_error()` call as normal.

If the `argument` parameter is provided, it must be one of
the parameters to the function, indicating the parameter that
is causing the error. The type checker may use this
information to produce a more precise error (for example, by
pointing the error caret at the specified argument in the
call site).

### is_provided(), is_positional(), and is_keyword()

These special functions have the following signatures:

    def is_provided(arg: Any, /) -> bool: ...
    def is_positional(arg: Any, /) -> bool: ...
    def is_keyword(arg: Any, /) -> bool: ...

`arg` must be one of the parameters to the function.
`is_provided()` returns True if the parameter was explicitly
provided in the call; that is, the default value was not
used. Similarly, `is_positional()` returns True if the
parameter was provided as a positional argument, and
`is_keyword()` returns True if the parameter was provided
as a keyword argument.

Parameters in Python can be provided in three ways,
which we call _argument kinds_ for the purpose of this
specification:

- `POSITIONAL`: at the call site, either a single
  positional argument or a variadic one (`*args`)
- `KEYWORD`: at the call site, either a sinngle keyword
  argument or a variadic one (`**kwargs`)
- `DEFAULT`: no value provided at the call site; the
  default defined in the function is used

Static analyzers must add a fourth kind in the presence
of calls with `*args` and `**kwargs`:

- `UNKNOWN`: the kind cannot be statically determined.
  This can happen in the following situations:
  - A positional-only parameter with a default in a call
    with `*args` of unknown size.
  - A keyword-only parameter with a default in a call
    with `**kwargs` of unknown size.
  - A positional-or-keyword parameter that matches either
    of the above conditions.
  - A positional-or-keyword parameter (with or without a
    default) in a call with both `*args` and `**kwargs`.

The three special functions map to these kinds as follows:

- `is_provided()`: kind is `POSITIONAL` or `KEYWORD`
- `is_positional()`: kind is `POSITIONAL`
- `is_keyword()`: kind is `KEYWORD`

Thus, there is no way to distinguish between `DEFAULT`
and `UNKNOWN`, and a parameter for which `is_provided()`
returns False in the type evaluator may actually be
provided at runtime.

For variadic parameters (`*args` and `**kwargs`), the
kind is either `DEFAULT` if no arguments are provided
to the parameter, or either `POSITIONAL` (for `*args`)
or `KEYWORD` (for `**kwargs`) if arguments may be
provided. If the type checker can
prove that a variadic argument is empty, `is_provided()`
may return False. (For example, given a definition
`def f(*args)` and a call `f(*())`, `is_provided(args)`
may return False.)

Examples:

    @evaluated
    def reject_arg(arg: int = 0) -> None:
        if is_provided(arg):
            show_error("error")

    args: Any = ...
    kwargs: Any = ...
    reject_arg()  # ok
    reject_arg(0)  # error
    reject_arg(arg=0)  # error
    reject_arg(*args)  # ok
    reject_arg(**kwargs)  # ok

    @evaluated
    def reject_star_args(*args: int) -> None:
        if is_provided(args):
            show_error("error")

    reject_star_args()  # ok
    reject_star_args(1)  # error
    reject_star_args(*(1,))  # error
    reject_star_args(*())  # may error, depending on type checker

    @evaluated
    def reject_star_kwargs(**kwargs: int) -> None:
        if is_provided(kwargs):
            show_error("error")

    reject_star_kwargs()  # ok
    reject_star_kwargs(x=1)  # error
    reject_star_kwargs(**{"x": 1})  # error
    reject_star_args(**{})  # may error, depending on type checker

    @evaluated
    def reject_keyword(arg: int = 0) -> None:
        if is_keyword(arg):
            show_error("error")

    reject_keyword()  # ok
    reject_keyword(0)  # ok
    reject_keyword(arg=0)  # error
    reject_keyword(*args)  # ok
    reject_keyword(**kwargs)  # ok

    @evaluated
    def reject_positional(arg: int = 0)-> None:
        if is_positional(arg):
            show_error("error")

    reject_keyword()  # ok
    reject_keyword(0)  # error
    reject_keyword(arg=0)  # ok
    reject_keyword(*args)  # ok
    reject_keyword(**kwargs)  # ok

    @evaluated
    def invalid(arg: object) -> None:
        if is_provided(x):  # error, not a function parameter
            show_error("error")

### is_of_type()

The special `is_of_type()` function has the following
signature:

    def is_oF_type(arg: object, type: Any, /, *, exclude_any: bool = True) -> bool: ...

`arg` must be one of the parameters to the function and
`type` must be a form that the type checker would accept
in a type annotation.

If `exclude_any` is False, `is_of_type(x, T)` returns true if `x` is
compatible with `T`; that is, if the type checker would
accept an assignment `_: T = x`.

If the `exclude_any` parameter is True (the default), normal type checking
rules are modified so that `Any` is no longer compatible with
any other type, but only with another `Any`. All other types
are still compatible with `Any`.

Examples:

    @evaluated
    def length_or_none(s: str | None = None):
        if is_of_type(s, str, exclude_any=False):
            return int
        else:
            return None

    any: Any = ...
    opt: int | None = ...
    reveal_type(length_or_none("x"))  # int
    reveal_type(length_or_none(None))  # None
    reveal_type(length_or_none(opt))  # int | None
    reveal_type(length_or_none(any))  # int

    @evaluated
    def length_or_none2(s: str | None):
        if is_of_type(s, str):
            return int
        elif is_of_type(s, None):
            return None
        else:
            return Any

    reveal_type(length_or_none2("x"))  # int
    reveal_type(length_or_none2(None))  # None
    reveal_type(length_or_none2(opt))  # int | None
    reveal_type(length_or_none2(any))  # Any

    @evaluated
    def nested_any(s: Sequence[Any]):
        if is_of_type(s, str):
            show_error("error")
        elif is_of_type(s, Sequence[str]):
            return str
        else:
            return int

    anyseq: Sequence[Any] = ...
    nested_any("x")  # error
    reveal_type(nested_any(["x"]))  # str
    reveal_type(nested_any([1]))  # int
    reveal_type(nested_any(any))  # int
    reveal_type(nested_any(anyseq))  # int

### Interaction with unions

Type checkers should apply normal type narrowing rules to arguments
that are of Union types. If only some members of a Union
match a condition, both branches of the conditional are
taken, with the parameter type narrowed appropriately in each
case. The return type of the function is the union of the two
branches.

For example:

    @evaluated
    def switch_types(arg: str | int):
        if is_of_type(arg, str):
            return int
        else:
            return str

    reveal_type(switch_types(1))  # str
    reveal_type(switch_types("x"))  # int
    union: int | str
    reveal_type(switch_types(union))  # int | str

### Generic evaluators

If any type variables appear in the parameters of the type evaluation
function, the type checker should first solve those and use the solution
in the body of the function:

    @evaluated
    def identity(x: T):
        return T

    reveal_type(evaluated(int()))  # int

As a result, `is_of_type()` checks that use a type variable work:

    @evaluated
    def safe_upcast(typ: Type[T1], value: object):
        if is_of_type(value, T1):
            return T1
        show_error("unsafe cast")
        return Any

    reveal_type(safe_upcast(object, 1))  # object
    reveal_type(safe_upcast(int, 1))  # int
    safe_upcast(str, 1)  # error

### Type compatibility

The type of an evaluated function is compatible with a
`Callable` with the same arguments and returning the
`Union` of the possible return types, and with any
`Callable` for which the evaluation function would
return a compatible type given the same arguments.

Examples:

    @evaluated
    def maybe_path(path: str | None):
        if path is None:
            return None
        else:
            return Path

    _: Callable[[str | None], Path | None] = maybe_path  # ok
    _: Callable[[None], None] = maybe_path  # ok
    _: Callable[[str], Path] = maybe_path  # ok
    _: Callable[[str | None], Path] = maybe_path  # error
    _: Callable[[str], Path | None] = maybe_path  # ok
    _: Callable[[Literal["x"]], Path] = maybe_path  # ok

### Runtime behavior

At runtime, the `@evaluated` decorator returns a dummy function
that throws an error when called, similar to `@overload`. In
order to support dynamic type checkers, it also stores the
original function, keyed by its fully qualified name.

A helper function is provided to retrieve all registered
evaluation functions for a given fully qualified name:

    def get_type_evaluations(
        fully_qualified_name: str
    ) -> Sequence[Callable[..., Any]]: ...

For example, if method `B.c` in module `a` has an evaluation function,
`get_type_evaluations("a.B.c")` will retrieve it.

Dummy implementations are provided for the various helper
functions (`is_provided()`, `is_positional()`, `is_keyword()`,
`is_of_type()`, and `show_error()`). These throw an error
if called at runtime.

The `reveal_type()` function has a runtime implementation
that simply returns its argument.

## Discussion

### Interaction with Any

The below is an evaluation function for a simplified
version of the `open()` builtin:

    @evaluated
    def open(mode: str):
        if is_of_type(mode, Literal["r", "w"]):
            return TextIO
        elif is_of_type(mode, Literal["rb", "wb"]):
            return BinaryIO
        else:
            return IO[Any]

What should `open()` return if the type of the `mode`
argument is `Any`? With the equivalent code expressed
using overloads, existing type checkers do not agree:
pyright picks the first overload that matches and returns
`int`, since `Any` is compatible with `None`; mypy and pyanalyze
see that multiple overloads might match and return `Any`.
There are good reasons for both choices,
as discussed [here](https://github.com/microsoft/pyright/issues/2521#issuecomment-956823577)
by Eric Traut. In particular, mypy's behavior is more sound
for a type checker, but pyright's behavior helps generate
better autocompletion suggestions in a language server.

Type evaluation functions potentially have
the same ambiguity, so in order to provide predictable
behavior across type checkers, we need to specify a single
behavior.

As specified above, our choice is to treat `Any` specially
by default within evaluation functions, making it
incompatible with other types, both within `is`/`==`
comparisons and within the `is_of_type` primitive.
This behavior makes it easiest
to write evaluation functions that read naturally and
behave as desired. In particular, this choice makes
`open(Any)` return `IO[Any]`, which is both the most
intuitive and the most useful result.

The most natural alternative is to make `is_of_type()` follow
normal type compatibility rules, where `Any` is compatible
with everything. But this would create confusing behavior
for evaluation functions like the one for `open()`:

- `open()` would return `TextIO` if `mode` is `Any`, which
  is too precise in general.
- The order of the `BinaryIO` and `TextIO` checks would
  matter:
  the function would behave differently if the two checks
  were flipped. This would be a subtle behavior that is
  not obvious to readers of the code.
- There would be no obvious way to provide a customized
  fallback behavior for `Any`. Technically, a check like
  `is_of_type(mode, Literal["r"]) and is_of_type(mode, Literal["w"])`
  could be used to check for `Any` (only `Any` is compatible
  with both literals), but this would be obscure and
  unreadable.
- It would be difficult to show an error for a particular
  parameter value. For example, a stub for `open()` might
  want to show a warning if the deprecated `rU` mode is used.
  The obvious way to do that would be to write
  `if mode == "rU": show_error(...)`, but if this returned
  true for `Any`, we would show the error for `mode: Any`.

As an additional example, consider functions that take
some object or `None` and return either `None` or a
transformed version of the object, like this:

    @evaluated
    def maybe_path(path: str | None):
        if path is None:
            return None
        else:
            return Path

Functions of this form are fairly common, and it is
natural to write them with the trivial branch (`None`) first,
both in the implementation and in the evaluation function.
But if `path is None` would be true for `Any`, the evaluation
function would return `None`, which is bad both for type
checkers and for autocomplete suggestions.

One downside of this behavior is that type checkers may
incorrectly flag `is None` checks after a `maybe_path()`
call as unreachable. However, such checks are usually only
enabled in a strict mode, and `Any` should be rare in
strictly typed code. Type checkers could also provide a
mechanism that labels types derived from an evaluation
function that used `Any` to disable diagnostics about
unreachable code.

Another alternative would be to use a mechanism similar to
mypy-style overload resolution: conditions that match due to
`Any` would essentially match neither branch and simply
return `Any`. This behavior would avoid returning any
overly precise types, but it would be useless for
autocompletion suggestions and would remove a lot of
useful type precision. For example, there would be no way
for the `open()` evaluation function to produce `IO[Any]`.

### Argument kind functions

The three argument kind functions `is_provided()`,
`is_positional()`, and `is_keyword()` are useful in various
ways:

- Functions implemented in C sometimes change behavior
  depending on the presence of an argument, without a
  meaningful default. For example, `dict.pop(key)` returns
  the key's value type (or else it raises an exception), but
  `dict.pop(key, default)` returns either the value type or
  the type of `default`. Currently overloads are necessary
  to represent this behavior, but `is_provided()` provides
  an alternative.
- It is common for new versions of Python to add or remove
  parameters. For example, `zip()` gained a `strict=` keyword
  argument in Python 3.10. Using `is_provided()` with a
  `sys.version_info` check, we can provide an error if the
  parameter is used in an older version, without duplicating
  the entire function definition.
- Similarly, new versions of Python often change parameters
  from positional-or-keyword to positional-only or vice
  versa. Version checks can be used with `is_positional()` or
  `is_keyword()` to reflect such changes in the stub.
- Library authors who want to evolve an API sometimes want
  to make a function parameter keyword-only. An evaluation
  function can be used to warn users who pass the parameter
  positionally without changing the runtime parameter kind,
  so that users have time to adapt before the runtime code
  is broken.

As an example, this is the current implemenation of `sum()`
in typeshed:

    if sys.version_info >= (3, 8):
        @overload
        def sum(__iterable: Iterable[_T]) -> _T | Literal[0]: ...
        @overload
        def sum(__iterable: Iterable[_T], start: _S) -> _T | _S: ...

    else:
        @overload
        def sum(__iterable: Iterable[_T]) -> _T | Literal[0]: ...
        @overload
        def sum(__iterable: Iterable[_T], __start: _S) -> _T | _S: ...

This is how it could be implemented using `@evaluated`:

    @evaluated
    def sum(__iterable: Iterable[_T], start: _S = ...):
        if not is_provided(start):
            return _T | Literal[0]
        if sys.version_info < (3, 8) and is_keyword(start):
            show_error("start is a positional-only argument in Python <3.8", argument=start)
        return _T | _S

### Generic evaluators

The specification for generic evaluators allows creating an evaluator
that checks whether two types have any overlap:

    T1 = TypeVar("T1")
    T2 = TypeVar("T2")

    @evaluated
    def safe_contains(elt: T1, container: Container[T2]) -> bool:
        if not is_of_type(elt, T2) and not is_of_type(container, Container[T1]):
            show_error("Element cannot be a member of container")

    lst: List[int]
    safe_contains("x", lst)  # error
    safe_contains(True, lst)  # ok (bool is a subclass of int)
    safe_contains(object(), lst)  # ok (List[int] is a subclass of Container[object])

Thus, type evaluation provides a way to implement checks similar to mypy's
[strict equality](https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-strict-equality)
flag directly in stubs.

## Compatibility

The proposal is fully backward compatible.

Type evaluation functions are going to be most frequently useful
in library stubs, where it is often important that multiple type
checkers can parse the stub. In order to unblock usage of the new
feature in stubs, type checker authors could simply ignore the
body of evaluation functions and rely on the signature. This would
still allow other type checkers to fully use the evaluation function.

## Possible extensions

The following features may be useful, but are deferred
for now for simplicity.

### Error categories

It may be useful to provide hints to the type checker
about the severity of a `show_error()` call. For example,
deprecation warnings could be marked so that the user can
control whether to show them.

One possibility is to add a keyword-only argument
`category: str = ...` to `show_error()`. We would specify
some standard categories that can be used in typeshed:

- `deprecation` (for deprecated behavior)
- `python_version` (for wrong Python version)
- `platform` (for wrong sys.platform)
- `warning` (for miscellaneous non-blocking issues)

Type checkers could add support for additional categories
as desired. Other type checkers would be expected to
silently ignore unrecognized category strings.

### Reusable error messages

Because `show_error()` requires a string literal as the
message, typeshed would contain a lot of hardcoded string
messages about version changes.

Some possible solutions include:

- Allow the message to be a variable of `Literal` type
  instead of a string literal. However, this would not
  allow customizing an error message to include e.g.
  the name of the argument or the Python version when
  some behavior changed.
- Allow the message to be a call to `.format()` on a
  string literal or `Literal` variable, where all the
  arguments are function arguments or literals:
  `show_error(NEW_IN_VERSION.format(arg, "3.10"))`.
- Allow the message to be a call to another evaluation
  function that returns a string literal instead of a type.
  This would allow even more complex logic for emitting the
  error message.

The last option could look like this:

    @evaluated
    def added_in_py_version(feature: str, version: str):
        return f"{feature} was added in Python {version}"

    def zip(strict: bool = False):
        if is_provided(strict) and sys.version_info < (3, 10):
            show_error(
                added_in_py_version("strict", "3.10"),
                argument="strict"
            )

### Adding attributes

A common pattern in type checker plugins is for the plugin
to add some extra attribute to the object. For example,
`@functools.total_ordering` inserts various dunder methods
into the class it decorates.

We could add an `add_attributes()` primitive that given
a type and a dictionary of attributes, modifies the type
to add these attributes.

Usage could look like this:

    @evaluated
    def total_ordering(cls: Type[T]):
        return add_attributes(
            cls,
            {"__eq__": Callable[[T, T], bool]}
        )

## Status

A partial implementation of this feature is available
in pyanalyze:

    from pyanalyze.extensions import evaluated, is_provided

    @evaluated
    def simple_evaluated(x: int, y: str = ""):
        if is_provided(y):
            return int
        else:
            return str

    def simple_evaluated(*args: object) -> Union[int, str]:
        if len(args) >= 2:
            return 1
        else:
            return "x"

Currently unsupported features include:

- Type compatibility for evaluated functions.
- Overloaded evaluated functions.

Areas that need more thought include:

- Interaction with overloads. It should be possible
  to register multiple evaluation functions for a
  function, treating them as overloads.
- Interaction with `__init__` and self types. How does
  an eval function set the self type of a function? Perhaps
  we can have the return type have special meaning just for
  `__init__` methods.
