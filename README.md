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

## Background

Pyanalyze is built on top of two lower-level abstractions: Python's built-in `ast` module and our own `node_visitor` abstraction, which is an extension of the `ast.NodeVisitor` class.

### Python AST module

The `ast` module (<https://docs.python.org/3/library/ast.html>) provides access to the abstract syntax tree (AST) of Python code. The AST is a tree-based representation of the structure of a Python program. For example, the string "`import a`" resolves into this AST:

```python
# ast.parse considers everything to be a module
Module(body=[
    # the module contains one statement of type Import
    Import(
        # names is a list; it would contain multiple elements for "import a, b"
        names=[
            alias(
                name='a',
                # if we did "import a as b", this would be "b" instead of None
                asname=None
            )
        ]
    )
])
```

The `ast.NodeVisitor` class provides a convenient way to run code that inspects an AST. For each AST node type, a NodeVisitor subclass can implement a method called `visit_<node type>`. When the visitor is run on an AST, this method will be called for each node of that type. For example, the following class could be used to find `import` statements:

```python
class ImportFinder(ast.NodeVisitor):
    def visit_Import(self, node):
        print("Found import statement: %s" % ast.dump(node))
```

### node_visitor.py

Pyanalyze uses an extension to `ast.NodeVisitor`, implemented in `pyanalyze/node_visitor.py`, that adds two main features: management of files to run the visitor on and management of errors that are found by the visitor.

The following is a simple example of a visitor using this abstraction---a visitor that will show an error for every `assert` and `import` statement found:

```python
import enum
from pyanalyze import node_visitor

class ErrorCode(enum.Enum):
    found_assert = 1
    found_import = 2

class BadStatementFinder(node_visitor.BaseNodeVisitor):
    error_code_enum = ErrorCode

    def visit_Assert(self, node):
        self.show_error(node, error_code=ErrorCode.found_assert)

    def visit_Import(self, node):
        self.show_error(node, error_code=ErrorCode.found_import)

if __name__ == '__main__':
    BadStatementFinder.main()
```

As an example, we'll run the visitor on a file containing this code:

```python
import a
assert True
```

Running the visitor without arguments gives the following output:

```
$ python example_visitor.py example.py
Error: found_import (code: found_import)
In example.py at line 1:
   1: import a
      ^
   2: assert True
   3:

Error: found_assert (code: found_assert)
In example.py at line 2:
   1: import a
   2: assert True
      ^
   3:
```

Using information stored in the node that caused the error, the `show_error` method finds the line and column in the Python source file where the error appears.

Passing an `error_code` argument to `show_error` makes it possible to conditionally suppress errors by passing a `--disable` command-line argument:

```
$ python example_visitor.py example.py --disable found_import
Error: found_assert (code: found_assert)
In example.py at line 2:
   1: import a
   2: assert True
      ^
   3:
```

Subclasses of `BaseNodeVisitor` can specify which errors are enabled by default by overriding `is_enabled_by_default` and the description shown for an error by overriding `get_description_for_error_code`.

## Design

Fundamentally, the way pyanalyze works is that it tries to infer, with as much precision as possible, what Python value or what kind of Python value each node in a file's AST corresponds to, and then uses that information to flag code that does something undesirable. Mostly, that involves identifying code that will cause the Python interpreter to throw an error at runtime, for example because it accesses an attribute that doesn't exist or because it passes incorrect arguments to a function. As much as possible, the script tries to evaluate whether an operation is allowed by asking Python whether it is: for example, whether the arguments to a function call are correct is decided by creating a function with the same arguments as the called function, calling it with the same arguments as in the call, and checking whether the call throws an error.

This is done by recursively visiting the AST of the file and building up a context of information gathered from previously visited nodes. For example, the `visit_ClassDef` method visits the body of the class within a context that indicates that AST nodes are part of the class, which enables method definitions within the class to infer the type of their `self` arguments as being the class. In some cases, the visitor will traverse the AST twice: once to collect places where names are set, and once again to check that every place a name is accessed is valid. This is necessary because functions may use names that are only defined later in the file.

### Name resolution

The name resolution component of pyanalyze makes it possible to connect usage of a Python variable with the place where it is defined.

Pyanalyze uses the `StackedScopes` class to simulate Python scoping rules. This class contains a stack of nested scopes, implemented as dictionaries, that contain names defined in a particular Python scope (e.g., a function). When the script needs to determine what a particular name refers to, it iterates through the scopes, starting at the top of the scope stack, until it finds a scope dictionary that contains the name. This is similar to how name lookup is implemented in Python itself. When a name that is accessed in Python code is not found in any scope object, pyanalyze will throw an error with code `undefined_name`.

When the script is run on a file, the scopes object is initialized with two scope levels containing builtin objects such as `len` and `Exception` and the file's module-level globals (found by importing the file and inspecting its `__dict__`). When it inspects the AST, it adds names that it finds in assignment context into the appropriate nested scope. For example, when the scripts sees a `FunctionDef` AST node, it adds a new function-level scope, and if the function contains a statement like `x = 1`, it will add the variable `x` to the function's scope. Then when the function accesses the variable `x`, the script can retrieve it from the function-level scope in the `StackedScopes` object.

The following scope types exist:

- `builtin_scope` is at the bottom of every scope stack and contains standard Python builtin objects.
- `module_scope` is always right above builtin_scope and contains module-global names, such as classes and functions defined at the global level in the file.
- `class_scope` is entered whenever the AST visitor encounters a class definition. It can contain nested class or function scopes.
- `function_scope` is entered for each function definition.

The function scope has a more complicated representation than the others so that it can reflect changes in values during the execution of a function. Broadly speaking, pyanalyze collects the places where every local variable is either written (definition nodes) or read (usage nodes), and it maps every usage node to the set of possible definition nodes that the value may come from. For example, if a variable is written to and then read on the next line, the usage node on the second line is mapped to the definition node on the first line only, but if a variable is set within both the if and the else branch of an if block, a usage after the if block will be mapped to definition nodes from both the if and the else block. If the variable is never set in some branches, a special marker object is used again, and pyanalyze will emit a `possibly_undefined_name` error.

Function scopes also support **constraints**. Constraints are restrictions on the values a local variable may take. For example, take the following code:

```python
def f(x: Union[int, None]) -> None:
    dump_value(x)  # Union[int, None]
    if x is not None:
        dump_value(x)  # int
```

In this code, the `x is not None` check is translated into a constraint that is stored in the local scope, similar to how assignments are stored. When a variable is used within the block, we look at active constraints to restrict the type. In this example, this makes pyanalyze able to understand that within the if block the type of `x` is `int`, not `Union[int, None]`.

The following constructs are understood as constraints:

- `if x is (not) None`
- `if (not) x`
- `if isinstance(x, <some type>)`
- `if issubclass(x, <some type>)`
- `if len(x) == <some value>`
- A function returning a `TypeGuard` or similar construct

Constraints are used to restrict the types of:

- Local variables
- Instance variables (e.g., after `if self.x is None`, the type of `self.x` is restricted)
- Nonlocal variables (variables defined in enclosing scopes)

### Type and value inference

Just knowing that a name has been defined doesn't tell what you can do with the value stored for the name. To get this information, each node visit method in `test_scope.py` can return an instance of the `Value` class representing the Python value that corresponds to the AST node. We also use type annotations in the code under consideration to get types for more values. Scope dictionaries also store `Value` instances to represent the values associated with names.

The following subclasses of `Value` exist:

- `AnyValue`, representing that the script knows nothing about the value a node can contain. For example, if a file contains only the function `def f(x): return x`, the name `x` will have an `AnyValue` as its value within the function, because there is no information to determine what value it can contain.
- `KnownValue` represents a value for which the script knows the concrete Python value. If a file contains the line `x = 1` and no other assignments to `x`, `x` will contain `KnownValue(1)`.
- `TypedValue` represents that the script knows the type but not the exact value. If the only assignment to `x` is a line `x = int(some_function())`, the script infers that `x` contains `TypedValue(int)`. More generally, the script infers any call to a class as resulting in an instance of that class. The type is also inferred for the `self` argument of methods, for comprehensions, for function arguments with type annotations, and in a few other cases. This class has several subtypes:
  - `NewTypeValue` corresponds to [`typing.NewType`](https://docs.python.org/3/library/typing.html#newtype); it indicates a distinct type that is identical to some other type at runtime. At Quora we use newtypes for helper types like `qtype.Uid`.
  - `GenericValue` corresponds to generics, like `List[int]`.
- `MultiValuedValue` indicates that multiple values are possible, for example because a variable is assigned to in multiple places. After the line `x = 1 if condition() else 'foo'`, `x` will contain `MultiValuedValue([KnownValue(1), KnownValue('foo')])`. This corresponds to [`typing.Union`](https://docs.python.org/3/library/typing.html#typing.Union).
- `UnboundMethodValue` indicates that the value is a method, but that we don't have the instance the method is bound to. This often comes up when a method in a class `SomeClass` contains code like `self.some_other_method`: we know that self is a `TypedValue(SomeClass)` and that `SomeClass` has a method `some_other_method`, but we don't have the instance that `self.some_other_method` will be bound to, so we can't resolve a `KnownValue` for it. Returning an `UnboundMethodValue` in this case makes it still possible to check whether the arguments to the method are valid.
- `ReferencingValue` represents a value that is a reference to a name in some other scopes. This is used to implement the `global` statement: `global x` creates a `ReferencingValue` referencing the `x` variable in the module scope. Assignments to it will affect the referenced value.
- `SubclassValue` represents a class object of a class or its subclass. For example, in a classmethod, the type of the `cls` argument is a `SubclassValue` of the class the classmethod is defined in. At runtime, it is either this class or a subclass.
- `NoReturnValue` indicates that a function will never return (e.g., because it always throws an error), corresponding to [`typing.NoReturn`](https://docs.python.org/3/library/typing.html#typing.NoReturn).

Each `Value` object has a method `can_assign` that checks whether types are correct. The call `X.can_assign(Y, ctx)` essentially answers the question: if we expect a value `X`, is it legal to pass a value `Y` instead? For example, `TypedValue(int).can_assign(KnownValue(1), ctx)` will succeed, because `1` is a valid `int`, but `TypedValue(int).can_assign(KnownValue("1"), ctx)` will fail, because `"1"` is not. In order to help with type checking generics, the return value of `can_assign` is a (possibly empty) dictionary of type variables mapping to their values.

### Call compatibility

When the visitor encounters a `Call` node (representing a function call) and it can resolve the object being called, it will check that the object can in fact be called and that it accepts the arguments given to it. This checks only the number of arguments and the names of keyword arguments, not their types.

The first step in implementing this check is to retrieve the signature for the callee. Python provides the `inspect.signature` function to do this, but for some callables additional logic is required. In addition, pyanalyze uses [typeshed](http://github.com/python/typeshed), a repository of types for standard library modules, to produce more precise signatures.

Once we have the signature, we can figure out whether the arguments passed to the callee in the AST node under consideration are compatible with the signature. This is done with the signature's `bind()` method. The abstraction also supports providing an _implementation function_ for a callable, a function that gets called with the types of the arguments to the function and that computes a more specific return type or checks the arguments.

### Non-existent object attributes

Python throws a runtime `AttributeError` when you try to access an object attribute that doesn't exist. Pyanalyze can statically find some kinds of code that will access non-existent attribute. The simpler case is when code accesses an attribute of a `KnownValue` , like in a file that has `import os` and then accesses `os.ptah`. In this case, we know the value that `os` contains, so we can try to access the attribute `ptah` on it, and show an error if the attribute lookup fails. Similarly, `os.path` will return a `KnownValue` of the `os.path` module, so that we can also check attribute lookups on `os.path`.

Another class of bugs involves objects accessing attributes on `self` that don't exist. For example, an object may set `self.promote` in its `__init__` method, but then access `self.promotion` in its `tree` method. To detect such cases, pyanalyze uses the `ClassAttributeChecker` class. This class keeps a record of every node where an attribute is written or read on a `TypedValue`. After checking all code that uses the class, it then takes the difference between the sets of read and written values and shows an error for every attribute that is read but never written. This approach is complicated by inheritance---subclasses may read values only written on the superclass, and vice versa. Therefore, the check doesn't trigger for any attribute that is set on any superclass or subclass of the class under consideration. It also doesn't trigger for any attributes of a class that has a base class that wasn't itself examined by the `ClassAttributeChecker`. This was needed to deal with Thrift classes that used attributes defined in superclasses outside of code checked by pyanalyze. Two superclasses are excluded from this, so that undefined attributes are flagged on their subclasses even though test_scope.py hasn't examined their definitions: `object` (the superclass of every class) and `qutils.webnode2.Component` (which doesn't define any attributes that are read by its subclasses).

### Finding unused code

Because pyanalyze tries to resolve all names and attribute lookups in code in a package, it was easy to extend it to determine which of the classes and functions defined in the package aren't accessed in any other code. This is done by recording every name and attribute lookup that results in a `KnownValue` containing a function or class defined in the package. After the AST visitor run, it compares the set of accessed objects with another set of all the functions and classes that are defined in submodules of the package. All objects that appear in the second set but not the first are probably unused. (There are false positives, such as functions that are registered in some registry by decorators, or those that are called from outside of `a` itself.) This check can be run by passing the `--find-unused` argument to pyanalyze.

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
