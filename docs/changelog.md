# Changelog

## Unreleased

- Check `with` and `async with` statements (#344)
- Allow storing type narrowing constraints in variables (#343)
- The first argument to `__new__` and `__init_subclass`
  does not need to be `self` (#342)
- Drop dependencies on `attrs` and `mypy_extensions` (#341)
- Correct location of error for incompatible parameter (#339)

## Version 0.5.0 (December 12, 2021)

- Recognize code following an infinite while loop as unreachable (#337)
- Recognize overloaded functions in stubs (#325)
- Fix handling of classes in stubs that have an incorrect `__qualname__`
  at runtime (#336)
- Fix type compatibility with generic functions (#335)
- Support function calls in annotations (#334)
- Better support for `TypeVar` bounds and constraints in stubs (#333)
- Improve type checking of `dict.update` and `dict.copy` (#328)
- Improve support for complex type aliases in stubs
  (#331)
- Limit special case for `Literal` callables to
  functions, not any callable (#329)
- Support for constants in stubs that do not exist
  at runtime (#330)
- Fix detection of PEP 604 union types in stubs (#327)
- Support literals over negative numbers in stubs
  and stringified annotations (#326)
- Improved overload matching algorithm (#321) (#324)
- Support runtime overloaded functions with `pyanalyze.extensions.overload` (#318)
- Internal support for overloaded functions (#316)
- Support `TypeVar` bounds and constraints (#315)
- Improve error messages involving concrete dictionary and sequence values (#312)
- More precise type inference for dict literals (#312)
- Support `AsynqCallable` with no arguments as an annotation (#314)
- Support iteration over old-style iterables providing only `__getitem__` (#313)
- Add support for runtime Protocols (#311)
- Stop inferring `Any` for non-runtime checkable Protocols on Python 3.6 and 3.7 (#310)
- Fix false positive where `multiprocessing.Pool.map_async`
  was identified as an asynq method (#306)
- Fix handling of nested classes (#305)
- Support Protocols for runtime types that are also defined in stubs (#297) (#307)
- Better detect signatures of methods in stub files (#304)
- Improve handling of positional-only arguments in stub files (#303)
- Fix bug where pyanalyze incorrectly inferred that an attribute always exists (#302)
- Fix compatibility of signatures with extra parameters (#301)
- Enhance `reveal_type()` output for `UnboundMethodValue` (#300)
- Fix handling of `async for` (#298)
- Add support for stub-only Protocols (#295)
- Basic support for stub-only types (#290)
- Require `typing_inspect>=0.7.0` (#290)
- Improve type checking of `raise` statements (#289)
- Support `Final` with arguments and `ClassVar` without arguments (#284)
- Add `pyanalyze.extensions.NoAny` (#283)
- Overhaul documentation (#282)
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
  - `DictIncompleteValue` now stores a sequence of `KVPair` object instead
    of just key-value pairs, enabling more granular information.
  - The type of a `TypedValue` may now be a string

## Version 0.4.0 (November 18, 2021)

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

## Version 0.3.1 (August 11, 2021)

- Exit with a non-zero exit code when errors occur
  (contributed by C.A.M. Gerlach)
- Type check the working directory if no command-line arguments
  are given (contributed by C.A.M. Gerlach)

## Version 0.3.0 (August 1, 2021)

- Type check calls on Unions properly
- Add `pyanalyze` executable
- Add `--enable-all` and `--disable-all` flags
  (contributed by C.A.M. Gerlach)
- Bug fixes

## Version 0.2.0 (May 17, 2021)

- Drop support for Python 2 and 3.5
- Improve unused object finder
- Add support for `TypeVar`
- Add support for `Callable`
- Add `pyanalyze.extensions`
- Add `pyanalyze.ast_annotator`
- Numerous other bug fixes and improvements

## Version 0.1.0 (May 29, 2020)

- Initial public release
