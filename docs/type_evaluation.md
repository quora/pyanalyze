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
    def round(number: SupportsRound[_T], ndigits: SupportsIndex | None = ...):
        if not is_set(ndigits) or ndigits is None:
            return int
        else:
            return _T

This makes it easier to see at a glance what the difference is between various overloads.

## Specification

Type-evaluated functions may be declared both at runtime and in stub files, similar to the existing `@overload` mechanism. At runtime, the evaluated function must immediately precede
the function implementation:

        @evaluated
        def round(number: SupportsRound[_T], ndigits: SupportsIndex | None = ...):
            if ...

        def round(number, ndigits=None):
            return number.__round__(ndigits)

In stubs, the implementation is omitted.

When a type checker encounters a call to a function for which a type evaluation has been provided, it should symbolically evaluate the body of the type evaluation until it reaches a `return` or `raise` statement. A `raise` statement indicates that the type checker should produce an error, a `return` statement provides the type that the call should return.

### Supported features

The body of a type evaluation uses a restricted subset of Python.
The only supported features are:

- `if` statements and `else` blocks. These can only contain conditions of the form specified below.
- `return` statements with return values that are interpretable as type annotations. This indicates the type that the function returns in a particular condition.
- `raise` statements of the form `raise Exception(message)`, where `message` is a string literal. When the code takes a branch that ends in a `raise` statement, the type checker should emit an error with the provided message.

Conditions in `if` statements may contain:
- Calls to the special `is_set()` function, which takes as its argument a variable. This functions returns True if the variable was provided as an argument in the call. For example, given a function `def f(arg=None): ...`, `is_set(arg)` would return True if the function is invoked as `f(None)`, but False if it is invoked as `f()`. A dummy runtime implementation of `is_set()` is provided.
- Calls to `isinstance()`, but with an extended meaning. `isinstance(arg, type)` will return True if and only if the type checker would accept an assignment `_: type = arg`. For example, `isinstance(arg, Literal["a", "b"])` is valid and returns True if the type of the argument is (for example) `Literal["a"]`.
- Expressions of the form `arg is (not) <constant>`, where `<constant>` may be True, False, or None.
- Expressions of the form `arg == <constant>` or `arg != <constant>`, where `<constant>` is any value valid inside `Literal[]` (a bool, int, string, or enum member).
- Version and platform checks that are otherwise valid in stubs, as specified in PEP 484.
- Multiple conditions combined with `and` or `or`.
- A negation of another condition with `not`.

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

### Interaction with unions

Type checkers should apply normal type narrowing rules to arguments
that are of Union types. For example, if the `ndigits` argument to
`round()` is of type `int | None`, the inferred return value should
be `_T | int`.

## Status

A partial implementation of this feature is available
in pyanalyze:

    from pyanalyze.extensions import evaluated, is_set

    @evaluated
    def simple_evaluated(x: int, y: str = ""):
        if is_set(y):
            return int
        else:
            return str

    def simple_evaluated(*args: object) -> Union[int, str]:
        if len(args) >= 2:
            return 1
        else:
            return "x"

Currently unsupported features include:
- Comparison against enum members
- Version and platform checks
- Use of `and` and `or`

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
