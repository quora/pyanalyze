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
  until it reaches a `return` or `raise` statement. A 
  `raise` statement indicates that the type checker should 
  produce an error, a `return` statement provides the type 
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
        raise Exception("error")

    x = always_errors(1)  # error
    reveal_type(x)  # Any

    @evaluated
    def always_errors_with_type(x: int) -> str:
        raise Exception("error")

    x = always_errors(1)  # error
    reveal_type(x)  # str

### Supported features

The body of a type evaluation uses a restricted subset of Python.
The only supported features are:

- `if` statements and `else` blocks. These can only contain conditions of the form specified below.
- `return` statements with return values that are interpretable as type annotations. This indicates the type that the function returns in a particular condition.
- `raise` statements of the form `raise Exception(message)`, where `message` is a string literal. When the code takes a branch that ends in a `raise` statement, the type checker should emit an error with the provided message. The return type of the call is the return annotation on the evaluation function, or `Any` if there is none.

Conditions in `if` statements may contain:
- A call to one of the following functions, which are covered
  in more detail below:
  - `is_provided()`, which returns whether a parameter was
    explicitly provided in a call.
  - `is_of_type()`, which returns whether a parameter is of
    a particular type.
- Expressions of the form `arg is (not) <constant>`, where `<constant>` may be True, False, or None. If `arg` is `Any`, the condition always matches.
- Expressions of the form `arg == <constant>` or `arg != <constant>`, where `<constant>` is any value valid inside `Literal[]` (a bool, int, string, or enum member). If `arg` is `Any`, the condition always matches.
- Version and platform checks that are otherwise valid in stubs, as specified in PEP 484.
- Multiple conditions combined with `and` or `or`.
- A negation of another condition with `not`.

### is_provided()

The special `is_provided()` function has the following
signature:

    def is_provided(arg: Any) -> bool: ...

`arg` must be one of the parameters to the function. The
function returns True if the parameter was explicitly
provided in the call; that is, the default value was not
used.

For variadic parameters (`*args` and `**kwargs`), the 
function returns True if any non-variadic arguments were 
passed that would go into these variadic parameters, or if
a variadic argument was passed. If the type checker can
prove that a variadic argument is empty, `is_provided()`
may return False. (For example, given a definition
`def f(*args)` and a call `f(*())`, `is_provided(args)`
may return False.)

It is an error to call `is_provided()` on an argument
that lacks a default and is not variadic.

Examples:

    @evaluated
    def reject_arg(arg: int = 0) -> None:
        if is_provided(arg):
            raise Exception("error")
    
    reject_arg()  # ok
    reject_arg(0)  # error
    reject_arg(arg=0)  # error

    @evaluated
    def reject_star_args(*args: int) -> None:
        if is_provided(args):
            raise Exception("error")

    reject_star_args()  # ok
    reject_star_args(1)  # error
    reject_star_args(*(1,))  # error
    reject_star_args(*())  # may error, depending on type checker

    @evaluated
    def reject_star_kwargs(**kwargs: int) -> None:
        if is_provided(kwargs):
            raise Exception("error")

    reject_star_kwargs()  # ok
    reject_star_kwargs(x=1)  # error
    reject_star_kwargs(**{"x": 1})  # error
    reject_star_args(**{})  # may error, depending on type checker

    @evaluated
    def invalid(arg: object) -> None:
        if is_provided(arg):  # error, cannot call is_provided() on required parameter
            raise Exception("error")
        if is_provided(x):  # error, not a function parameter
            raise Exception("error")


### is_of_type()

The special `is_of_type()` function has the following
signature:

    def is_oF_type(arg: object, type: Any, *, exclude_any: bool = False) -> bool: ...

`arg` must be one of the parameters to the function and
`type` must be a form that the type checker would accept
in a type annotation.

By default, `is_of_type(x, T)` returns true if `x` is
compatible with `T`; that is, if the type checker would
accept an assignment `_: T = x`.

If the `exclude_any` parameter is True, normal type checking
rules are modified so that `Any` is no longer compatible with
any other type, but only with another `Any`.

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
            raise Exception("error")
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


### Interaction with Any

What should `round()` return if the type of the `ndigits`
argument is `Any`? Existing type checkers do not agree:
pyright picks the first overload that matches and returns
`int`, since `Any` is compatible with `None`; mypy and pyanalyze
see that multiple overloads might match and return `Any`. There
are good reasons for both choices<!-- insert link to Eric's explanation-->,
and we allow the same behavior for type evaluations.

Type checkers should pick one of the following two behaviors and 
document their choice:
1. All checks (`isinstance`, `is`, `==`) against variables typed 
   as `Any` in the body of type evaluation succeed. 
   `round(..., Any)` returns `int`. Note that
   this means that switching the `if` and `else` blocks may change
   visible behavior.
2. Conditions on variables typed as `Any` take both branches of the
   conditional. If the two branches return different types, `Any`
   is returned instead. `round(..., Any)` returns `Any`.

Motivating example:

    @evaluated
    def open(mode: str):
        if is_of_type(mode, Literal["r", "w"], exclude_any=True):
            return TextIO
        elif is_of_type(mode, Literal["rb", "wb"], exclude_any=True):
            return BinaryIO
        else:
            return IO[Any]
    
    any: Any = ...
    reveal_type(open("r"))  # TextIO
    reveal_type(open("rb"))  # BinaryIO
    reveal_type(open(any))  # IO[Any]

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
            return Path(path)
    
    _: Callable[[str | None], Path | None] = maybe_path  # ok
    _: Callable[[None], None] = maybe_path  # ok
    _: Callable[[str], Path] = maybe_path  # ok
    _: Callable[[str | None], Path] = maybe_path  # error
    _: Callable[[str], Path | None] = maybe_path  # ok
    _: Callable[[Literal["x"]], Path] = maybe_path  # ok

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
- `is_set()` and `is_provided()` as now specified
- Comparison against enum members
- Version and platform checks
- Use of `and` and `or`
- Usage in stubs
- pyanalyze should provide a way to register
  an evaluation function for a runtime function,
  to replace some impls.
- Type compatibility for evaluated functions.
- Checking evaluation functions for correctness.

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
