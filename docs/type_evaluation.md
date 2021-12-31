# Type evaluation

Type evaluation is a mechanism for replacing complex
overloads and version checks. For example, consider the
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
  that the call should return.
- If execution reached a `return` statement, return the type
  provided by that statement. Otherwise, return the type
  set in the evaluation function's return annotation, or
  `Any` if there is no return annotation.

Type checkers are encouraged to provide a strictness option
that produces an error if an evaluation function is missing
a type annotation on a parameter or return type. However, no
error should be provided if the return annotation is missing
and all branches (including error branches) return a type.

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

### Supported features

The body of a type evaluation uses a restricted subset of Python.
The only supported features are:

- `if` statements and `else` blocks. These can only contain conditions of the form specified below.
- `return` statements with return values that are interpretable as type annotations. This indicates the type that the function returns in a particular condition.
- `pass` statements, which do nothing.
- Calls to `show_error()`, which cause the type checker
  to emit an error. These are discussed further below.

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
specificatioon:

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
        if is_of_type(s, str):
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
        if is_of_type(s, str, exclude_any=True):
            return int
        elif is_of_type(s, None, exclude_any=True):
            return None
        else:
            return Any

    reveal_type(length_or_none2("x"))  # int
    reveal_type(length_or_none2(None))  # None
    reveal_type(length_or_none2(opt))  # int | None
    reveal_type(length_or_none2(any))  # Any

    @evaluated
    def nested_any(s: Sequence[Any]):
        if is_of_type(s, str, exclude_any=True):
            show_error("error")
        elif is_of_type(s, Sequence[str], exclude_any=True):
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

For example, if the `ndigits` argument to
`round()` is of type `int | None`, the inferred return value should
be `_T | int`.

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

- Usage in stubs
- pyanalyze should provide a way to register
  an evaluation function for a runtime function,
  to replace some impls.
- Type compatibility for evaluated functions.

Areas that need more thought include:

- Interaction with typevars
- Interaction with overloads. It should be possible
  to register multiple evaluation functions for a
  function, treating them as overloads.
- Consider adding support for `assert` and an
  ergonomical way to produce a standardized error
  if something is not supported in the current
  version or platform.
- Guidance on what the return annotation of an
  evaluation function should be. Most likely,
  it is treated as the default return type if
  execution reaches the end of the evaluation
  function. It can be omitted if the evaluation
  function always return.
- Add a `warn()` mechanism to warn on particular
  invocations. This can be useful as a mechanism
  to produce deprecation warnings.

Motivations can include:

- Less repetitive overload writing
- Ability to customize error messages
- Potential for additional features that work
  across type checkers
