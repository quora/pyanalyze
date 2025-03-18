# Implementation

Implementing features of the type system often involves a lot of
choices and research. This page contains notes about many features
of the type system and how they are implemented in pyanalyze.

## typing

First, let's go over the primitives in
[the typing docs](https://docs.python.org/3.11/library/typing.html#special-typing-primitives).

### Any

`Any` is simple: it is compatible with any type, and any type is
compatible with it. In pyanalyze's type system, it is represented
by `AnyValue`. Instances of `AnyValue` have a `source` parameter,
which indicates where the `Any` came from. Common sources include
`explicit` (when the user explicitly wrote `Any`), `unannotated`
(an unannotated parameter), and `generic_argument` (a missing type
parameter to a generic). Currently the distinction is not used in
too many places, but it helps understand better where an `Any`
comes from.

This concept was inspired by mypy's `TypeOfAny` enum. Pyright similarly
makes a distinction between `Unknown` and `Any`, but I haven't
studied it in detail.

Other type checkers also have the concept of a type that inherits
from `Any`. This is used in the typeshed stubs, for example, for
`Mock`. Pyanalyze doesn't currently support this and instead
special cases `Mock`.

### Never and NoReturn

Both of these represent the bottom type. `NoReturn` has been the
traditional name since early typing, but in 3.11 we are adding the
`Never` name for clarity. Pyanalyze treats both exactly the same.

Pyanalyze represents `Never`
as an empty union, `MultiValuedValue([])`, but there is special
casing in place so that there is only ever one instance of it,
which is `NO_RETURN_VALUE`.

No type is assignable to `Never`, except for `Any`. Both mypy and
pyright agree with this. However, `Never` is assignable to every
type. This is consistent with type theory, and it is useful in
practice because it means that in a function that deals with a
value that is supposedly of type `Never`, you get to do something
with it (e.g., raise a runtime error).

If a function call (including an implicit function call in an
operator) or `await` expression evaluates to `Never`,
pyanalyze infers that execution ends, just as if there was a
`return` statement. It would make sense to do this for all
expressions, so perhaps I should try that. Looking at the
code, it seems like there may be some complexity around asynq
functions.

### Self

`Self` was introduced by [PEP 673](https://www.python.org/dev/peps/pep-0673/)
to represent the type of the current class. Pyanalyze's implementation
turns `Self` into a single, global type variable. When attributes
are resolved, we run a `TypeVar` substitution pass that replaces `Self`
with the type of the owning class.

This works well for normal use of `Self`, but may not work correctly
in other contexts, such as when checking for compatible overrides.

### TypeAlias

`TypeAlias` explicitly marks a type alias. It is simple to support
and not especially necessary for pyanalyze, since we don't make a
strong distinction between types and values in the first place.

### Tuple

The `Tuple` special form corresponds to the internal
`SequenceIncompleteValue` class. See under "Sequence objects" for more.

### Union

The internal concept corresponding to `Union` is the awkwardly named
`MultiValuedValue` class. As with the runtime `Union`, nested unions
are flattened; the invariant is that no member of the union is itself
a union (although it may be a union inside of `Annotated`).

Pyanalyze tends to infer a lot of big
union types (for example, iterating over a list of literals will
yield a value typed as the union of all of these literals). This has
led to some performance issues in the past, and to mitigate that
the `MultiValuedValue` class has a special case for sets of hashable
literals.

### Optional

`Optional[T]` is exactly equivalent to `Union[T, None]`, and pyanalyze
has no special treatment for `Optional`.

### Callable

`Callable` is represented internally as `CallableValue`, which itself
is a thin wrapper around the `Signature` object. This object is inspired
by `inspect.Signature`, but supports additional parameter kinds specific
to static typing.

<!--
Other things to cover:
- What are these kinds? (PARAM_SPEC, ELLIPSIS)
- Argument binding. TypedDict in arguments. Unused *args.
- Compatibilty between callables
-->

### Concatenate

`Concatenate` is relatively simple to support because of the way `ParamSpec`
is implemented: a `Callable` with a `Concatenate` argument list is represented
by a `Signature` with a normal parameter followed by a parameter of kind
`PARAM_SPEC`.

### Type

PEP 484 specifies that `Type[]` only accepts a limited set of type
parameters:

- `Type[Any]`, which pyanalyze treats as exactly equivalent to plain `type`
- `Type[ConcreteClass]`, interpreted as `SubclassValue(TypedValue(ConcreteClass))`
- `Type[T]`, interpreted as `SubclassValue(TypeVarValue(T))`

Pyanalyze also supports `Type[Union[A, B]]`, and treats it as equivalent to
`Union[Type[A], Type[B]]`. A strict reading of PEP 484 does not allow this.
We also allow `Type[List[int]]`, though I'm not sure how meaningful that is.

Pyanalyze also allows a `TypedDict` type as the argument to `Type[]`, though
strictly speaking it should not. Pyright allows this too but mypy rejects it.

### Literal

Literal types (represented by `KnownValue`) are very prominent in pyanalyze's
type system: it infers literal types not only for the types allowed by PEP 586
(ints, strings, bytes objects, bools, enums, and `None`), but also for all other
types. For example, the `sys` module object would be represented as
`KnownValue(sys)`.

Inferring literal types is tricky for other type checkers because of variance.
The fact that pyanalyze mostly ignores variance helps here.

There is one special case for type compatibility: if the argument to `Literal[]`
is a function object, all compatible callables are accepted. This is similar to
[a rejected proposal](https://peps.python.org/pep-0677/#functions-as-types)
from PEP 677. I have not found this feature useful in practice very often.

### ClassVar

Pyanalyze mostly ignores `ClassVar` at the moment. The main effect it has on a
type checker is to disallow attribute assignment on instances, but at the moment
pyanalyze does not do much checking of assignments.

### Final

`Final` disallows assignment to a variable or attribute. Pyanalyze does not
currently enforce this.

### Annotated

### TypeGuard

### Generic

### TypeVar

### ParamSpec

### AnyStr

### Protocol

### runtime_checkable

### NamedTuple

### NewType

### TypedDict

## Language features

### Sequence objects

### Binary operators

### Comparison operators


