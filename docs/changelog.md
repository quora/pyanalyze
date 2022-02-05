# Changelog

## Unreleased

- Support `__init__` and `__new__` signatures from typeshed (#429)
- Add plugin providing a precise type for `dict.get` calls (#460)
- Fix internal error when an `__eq__` method throws (#461)
- Fix handling of `async def` methods in stubs (#459)
- Treat Thrift enums as compatible with protocols that
  `int` is compatible with (#457)
- Assume that dataclasses have no dynamic attributes (#456)
- Treat Thrift enums as compatible with `int` (#455)
- Fix treatment of `TypeVar` with bounds or constraints
  as callables (#454)
- Improve `TypeVar` solution algorithm (#453)
- Cache decisions about whether classes implement protocols (#450)
- Fix application of multiple suggested changes per file
  when an earlier change has added or removed lines (#449)
- Treat `NoReturn` like `Any` in `**kwargs` calls (#446)
- Improve error messages for overloaded calls (#445)
- Infer `NoReturn` instead of `Any` for unreachable code (#443)
- Make `NoReturn` compatible with all other types (#442)
- Fix treatment of walrus operator in `and`, `or`, and `if/else`
  expressions (#441)
- Refactor `isinstance()` support (#440)
- Exclude `Any[unreachable]` from unified values (#439)
- Add support for `reveal_locals()` (#436)
- Add support for `assert_error()` (#435)
- Add support for `assert_type()` (#434)
- `reveal_type()` and `dump_value()` now return their argument,
  the anticipated behavior for `typing.reveal_type()` in Python
  3.11 (#433)
- Fix return type of async generator functions (#431)
- Type check function decorators (#428)
- Handle `NoReturn` in `async def` functions (#427)
- Support PEP 673 (`typing_extensions.Self`) (#423)
- Updates for compatibility with recent changes in typeshed (#421):
  - Fix override compatibility check for unknown callables 
  - Fix usage of removed type `_typeshed.SupportsLessThan`
- Remove old configuration abstraction (#414)

## Version 0.6.0 (January 12, 2022)

Release highlights:
- Support for configuration through `pyproject.toml`. The old
  configuration mechanism will be removed in the next release.
- Support for experimental new type evaluation mechanism, providing
  a more powerful replacement for overloads.
- Support for suggesting annotations for unannotated code.

Full changelog:
- Support generic type evaluators (#409)
- Implement return annotation behavior for type evaluation
  functions (#408)
- Support `extend_config` option in `pyproject.toml` (#407)
- Remove the old method return type check. Use the new
  `incompatible_override` check instead (#404)
- Migrate remaining config options to new abstraction (#403)
- Fix stub classes with references to themselves in their
  base classes, such as `os._ScandirIterator` in typeshed (#402)
- Fix type narrowing on the `else` case of `issubclass()`
  (#401)
- Fix indexing a list with an index typed as a
  `TypeVar` (#400)
- Fix "This function should have an @asynq() decorator"
  false positive on lambdas (#399)
- Fix compatibility between Union and Annotated (#397)
- Fix potential incorrect inferred return value for
  unannotated functions (#396)
- Fix compatibility between Thrift enums and TypeVars (#394)
- Fix accessing attributes on Unions nested within
  Annotated (#393)
- Fix interaction of `register_error_code()` with new
  configuration mechanism (#391)
- Check against invalid `Signature` objects and prepare
  for refactoring `Signature` compatibility logic (#390)
- Treat `int` and `float` as compatible with `complex`,
  as specified in PEP 484 (#389)
- Do not error on boolean operations on values typed
  as `object` (#388)
- Support type narrowing on enum types and `bool`
  in `match` statements (#387)
- Support some imports from stub-only modules (#386)
- Support type evaluation functions in stubs (#386)
- Support `TypedDict` in stubs (#386)
- Support `TypeAlias` (PEP 612) (#386)
- Small improvements to `ParamSpec` support (#385)
- Allow `CustomCheck` to customize what values
  a value can be assigned to (#383)
- Fix incorrect inference of `self` argument on
  some nested methods (#382)
- Fix compatibility between `Callable` and `Annotated`
  (#381)
- Fix inference for nested `async def` functions (#380)
- Fix usage of type variables in function parameters
  with defaults (#378)
- Support the Python 3.10 `match` statement (#376)
- Support the walrus (`:=`) operator (#375)
- Initial support for proposed new "type evaluation"
  mechanism (#374, #379, #384, #410)
- Create command-line options for each config option (#373)
- Overhaul treatment of function definitions (#372)
  - Support positional-only arguments
  - Infer more precise types for lambda functions
  - Infer more precise types for nested functions
  - Refactor related code
- Add check for incompatible overrides in child classes
  (#371)
- Add `pyanalyze.extensions.NoReturnGuard` (#370)
- Infer call signatures for `Type[X]` (#369)
- Support configuration in a `pyproject.toml` file (#368)
- Require `typeshed_client` 2.0 (#361)
- Add JSON output for integrating pyanalyze's output with other
  tools (#360)
- Add check that suggests parameter and return types for untyped
  functions, using the new `suggested_parameter_type` and
  `suggested_return_type` codes (#358, #359, #364)
- Extract constraints from multi-comparisons (`a < b < c`) (#354)
- Support positional-only arguments with the `__` prefix
  outside of stubs (#353)
- Add basic support for `ParamSpec` (#352)
- Fix error on use of `AbstractAsyncContextManager` (#350)
- Check `with` and `async with` statements (#344)
- Improve type compatibility between generics and literals (#346)
- Infer signatures for method wrapper objects (bound methods
  of builtin types) (#345)
- Allow storing type narrowing constraints in variables (#343)
- The first argument to `__new__` and `__init_subclass__`
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
