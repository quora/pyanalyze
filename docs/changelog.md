# Changelog

## Unreleased

- More PEP 695 support: generic classes and functions. Scoping rules
  are not yet fully implemented. (#703)
- Fix type inference when constructing user-defined generic classes
  (#703)
- Fix bug in type compatibility check between known-length sequence
  types and literal values (#701)
- Fix Protocol compatibility issue with Python 3.13 and an upcoming
  release of typing-extensions (#716)

## Version 0.11.0 (October 3, 2023)

- Partial support for PEP 695-style type aliases. Scoping changes
  are not yet fully implemented. (#690, #692)
- Fix tests to account for new `typeshed_client` release
  (#694)
- Add option to disable all error codes (#659)
- Add hacky fix for bugs with hashability on type objects (#689)
- Show an error on calls to `typing.Any` (#688)
- Add command-line option `-c`/`--code` to typecheck code from
  the command line (#685)
- Add a `pyanalyze.extensions.EnumName` predicate and infer it
  as the value for the `.name` attribute on enums. Also fix
  type inference for enum "properties" on Python 3.11 and up. (#682)
- Allow `pyanalyze.runtime.is_compatible` to be used to narrow
  types (#681, #687)
- Fix usage of `assert_type()` with `Any` and with unions of
  `Annotated` objects (#680)
- Support inferring `MinLen` and `MaxLen` annotations based
  on `len()` checks (#680)
- Expose a convenience API for runtime type checking in the
  `pyanalyze.runtime` module (#674)
- Support for annotations from the `annotated-types` library (#673)
- Detect undefined attributes on Pydantic models (#670)
- Remove duplicate "attribute_is_never_set" error for classes
  with predefined attributes (#670)
- Add hook for overriding the value inferred for attributes on
  literals (#669)
- Support classes that set a `__signature__` attribute to
  define their constructor signature (such as Pydantic models) (#665)
- Declare support for Python 3.12. Not all new features in
  Python 3.12 are supported yet. (#656)
- Fix treatment of `@property` by the `incompatible_override`
  check (#653)
- Drop support for Python 3.7 (#654)
- Add hardcoded support for `pytest.raises` to avoid false
  positives (#651)
- Fix crash with nested classes in stubs. For now, `Any` is
  inferred for such classes (#647)
- Add `disallowed_imports` configuration option to disallow
  imports of specific modules (#645, #646)
- Consider an annotated assignment without a value to be
  an exported name (#644)
- Improve the location where `missing_parameter_annotation`
  errors are reported (#643)
- Add support for suppressing errors in blocks based on
  `sys.platform` and `sys.version_info` checks (#641)
- Fix compatibility between stub-only callable classes
  and the bare `Callable` annotation (#640)
- Add new error code `missing_generic_parameters` (off by
  default) that flags missing parameters to generic types
  such as `list` (#637)
- Add new error code `reveal_type` for `reveal_type()`
  and similar functions, which previously emitted
  `inference_failure` (#636)
- Take into account additional base classes declared in stub
  files (fixing some false positives around `typing.IO`) (#635, #639)
- Fix crash on stubs that contain dict or set literals (#634)
- Remove more old special cases and improve robustness of
  annotation parsing (#630)
- Remove dependency on `typing_inspect` (#629)
- Fix use of `Literal` types with `typing_extensions` 4.6.0 (#628)

## Version 0.10.1 (May 22, 2023)

- Fix errors with protocol matching on `typing_extensions` 4.6.0
  (#626)
- Fix false positive error when annotations refer to classes defined
  inside functions (#624)

## Version 0.10.0 (May 10, 2023)

- Infer the signature for built-in static methods, such as `dict.fromkeys` (#619)
- Fix type inference for subscripting on `Sequence` (#618)
- Improve support for Cythonized methods (#617)
- Add support for the PEP 698 `@override` decorator (#614)
- Add support for `__new__` methods returning `typing.Self`, fixing
  various failures with the latest release of `typeshed-client` (#615)
- Add support for importing stub-only modules in other stubs (#615)
- Fix signature compatibility bug involving `**kwargs` and positional-only
  arguments (#615)
- Fix type narrowing with `in` on enum types in the negative case (#606)
- Fix crash when `getattr()` on a module object throws an error (#603)
- Fix handling of positional-only arguments using `/` syntax in stubs (#601)
- Fix bug where objects with a `__call__` method that takes `*args` instead
  of `self` were not considered callable (#600)
- Better typechecking support for async generators (#594)

## Version 0.9.0 (January 16, 2023)

Release highlights:

- Support for PEP 702 (`@typing.deprecated`) (#578)
- Add experimental `@has_extra_keys` decorator for `TypedDict` types
- Support more Python 3.11 features (`except*` and `get_overloads`)

Full changelog:

- Support `typing_extensions.get_overloads` and `typing.get_overloads` (#589)
- Support `in` on objects with only `__iter__` (#588)
- Do not call `.mro()` method on non-types (#587)
- Add `class_attribute_transformers` hook (#585)
- Support for PEP 702 (`@typing.deprecated`) (#578)
- Simplify import handling; stop trying to import modules at type checking time (#566)
- Suggest using keyword arguments on calls with too many positional arguments (#572)
- Emit an error for unknown `TypedDict` keys (#567)
- Improve type inference for f-strings containing literals (#571)
- Add experimental `@has_extra_keys` decorator for `TypedDict` types (#568)
- Fix crash on recursive type aliases. Recursive type aliases now fall back to `Any` (#565)
- Support `in` on objects with only `__getitem__` (#564)
- Add support for `except*` (PEP 654) (#562)
- Add type inference support for more constructs in `except` and `except*` (#562)

## Version 0.8.0 (November 5, 2022)

Release highlights:

- Support for Python 3.11
- Drop support for Python 3.6
- Support for PEP 692 (`Unpack` on `**kwargs`)

Full changelog:

- Infer `async def` functions as returning `Coroutine`, not
  `Awaitable` (#557, #559)
- Drop support for Python 3.6 (#554)
- Require `typeshed_client>=2.1.0`. Older versions will throw
  false-positive errors around context managers when
  `typeshed_client` 2.1.0 is installed. (#554)
- Fix false positive error certain method calls on literals (#548)
- Preserve `Annotated` annotations on access to methods of
  literals (#541)
- `allow_call` callables are now also called if the arguments
  are literals wrapped in `Annotated` (#540)
- Support Python 3.11 (#537)
- Fix type checking of binary operators involving unions (#531)
- Improve `TypeVar` solution heuristic for constrained
  typevars with multiple solutions (#532)
- Fix resolution of stringified annotations in `__init__` methods (#530)
- Type check `yield`, `yield from`, and `return` nodes in generators (#529)
- Type check calls to comparison operators (#527)
- Retrieve attributes from stubs even when a runtime
  equivalent exists (#526)
- Fix attribute access to stub-only names (#525)
- Remove a number of unnecessary special-cased signatures
  (#499)
- Add support for use of the `Unpack` operator to
  annotate heterogeneous `*args` and `**kwargs` parameters (#523)
- Detect incompatible types for some calls to `list.append`,
  `list.extend`, `list.__add__`, and `set.add` (#522)
- Optimize local variables with very complex inferred types (#521)

## Version 0.7.0 (April 13, 2022)

Release highlights:

- Support for PEP 673 (`Self`)
- Support for PEP 675 (`LiteralString`)
- Support for `assert_type` and other additions to `typing` in Python 3.11

Full changelog:

- Remove `SequenceIncompleteValue` (#519)
- Add implementation function for `dict.pop` (#517)
- Remove `WeakExtension` (#517)
- Fix propagation of no-return-unless constraints from calls
  to unions (#518)
- Initial support for variable-length heterogeneous sequences
  (required for PEP 646). More precise types are now inferred
  for heterogeneous sequences containing variable-length
  objects. (#515, #516)
- Support `LiteralString` (PEP 675) (#514)
- Add `unused_assignment` error code, separated out from
  `unused_variable`. Enable these error codes and
  `possibly_undefined_name` by default (#511)
- Fix handling of overloaded methods called on literals (#513)
- Partial support for running on Python 3.11 (#512)
- Basic support for checking `Final` and for checking re-assignments
  to variables declared with a specific type (#505)
- Correctly check the `self` argument to `@property` getters (#506)
- Correctly track assignments of variables inside `try` blocks
  and inside `with` blocks that may suppress exceptions (#504)
- Support mappings that do not inherit from `collections.abc.Mapping`
  (#501)
- Improve type inference for calls to `set()`, `list()`, and
  `tuple()` with union arguments (#500)
- Remove special-cased signatured for `sorted()` (#498)
- Support type narrowing on `bool()` calls (#497)
- Support context managers that may suppress exceptions (#496)
- Fix type inference for `with` assignment targets on
  Python 3.7 and higher (#495)
- Fix bug where code after a `while` loop is considered
  unreachable if all `break` statements are inside of `if`
  statements (#494)
- Remove support for detecting properties that represent
  synchronous equivalents of asynq methods (#493)
- Enable exhaustive checking of enums and booleans (#492)
- Fix type narrowing in else branch if constraint is stored in a
  variable (#491)
- Fix incorrectly inferred `Never` return type for some function
  implementations (#490)
- Infer precise call signatures for `TypedDict` types (#487)
- Add mechanism to prevent crashes on objects
  with unusual `__getattr__` methods (#486)
- Infer callable signatures for objects with a
  `__getattr__` method (#485, #488)
- Do not treat attributes that raise an exception on access
  as nonexistent (#481)
- Improve detection of unhashable dict keys and set members (#469)
- The `in` and `not in` operators always return
  booleans (#480)
- Allow `NotImplemented` to be returned from special
  methods that support it (#479)
- Fix bug affecting type compatibility between
  generics and literals (#474)
- Add support for `typing.Never` and `typing_extensions.Never` (#472)
- Add `inferred_any`, an extremely noisy error code
  that triggers whenever the type checker infers something as `Any` (#471)
- Optimize type compatibility checks on large unions (#469)
- Detect incorrect key types passed to `dict.__getitem__` (#468)
- Pick up the signature of `open()` from typeshed correctly (#463)
- Do not strip away generic parameters explicitly set to
  `Any` (#467)
- Fix bug that led to some overloaded calls incorrectly
  resolving to `Any` (#462)
- Support `__init__` and `__new__` signatures from typeshed (#430)
- Fix incorrect type inferred for indexing operations on
  subclasses of `list` and `tuple` (#461)
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
