pyanalyze
=========

Pyanalyze is a tool for programmatically detecting common mistakes in Python code, such as references to undefined variables and some categories of type mismatches.
It can be extended to add additional rules and perform checks specific to particular functions.

Some use cases for this tool include:

-   **Catching bugs before they reach production**. The script will catch accidental mistakes like writing "`collections.defalutdict`" instead of "`collections.defaultdict`", so that they won't cause errors in production. Other categories of bugs it can find include variables that may be undefined at runtime, duplicate keys in dict literals, and missing `await` keywords.
-   **Making refactoring easier**. When you make a change like removing an object attribute or moving a class from one file to another, pyanalyze will often be able to flag code that you forgot to change.
-   **Finding dead code**. It has an option for finding Python objects (functions and classes) that are not used anywhere in the codebase.
-   **Checking type annotations**. Type annotations are useful as documentation for readers of code, but only when they are actually correct. Although pyanalyze does not support the full Python type system (see [below](#type-system) for details), it can often detect incorrect type annotations.

Usage
-----

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
        ErrorCode.condition_always_true,
        ErrorCode.possibly_undefined_name,
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
from ast import AST
from pyanalyze.arg_spec import ExtendedArgSpec, Parameter
from pyanalyze.error_code import ErrorCode
from pyanalyze.name_check_visitor import NameCheckVisitor
from pyanalyze.value import KnownValue, TypedValue, Value
from typing import Dict

from database import run_query, parse_sql

def run_query_impl(
  variables: Dict[str, Value],  # parameters passed to the function
  visitor: NameCheckVisitor,   # can be used to show errors or look up names
  node: AST,  # for showing errors
) -> Value:
  sql = variables["sql"]
  if not isinstance(sql, KnownValue) or not isinstance(sql.val, str):
      visitor.show_error(
          node,
          "Argument to run_query() must be a string literal",
          error_code=ErrorCode.incompatible_call,
      )
      return

  try:
      parsed = parse_sql(sql)
  except ValueError as e:
      visitor.show_error(
          node,
          f"Invalid sql passed to run_query(): {e}",
          error_code=ErrorCode.incompatible_call,
      )
      return

  # check that the parsed SQL is valid...

  # pyanalyze will use this as the inferred return type for the function
  return TypedValue(list)

  class Config(pyanalyze.config.Config):
      def get_known_argspecs(self, arg_spec_cache):
          return {
              # This infers the parameter types and names from the function signature
              run_query: arg_spec_cache.get_argspec(
                  run_query, implementation=run_query_impl,
              )
              # You can also write the signature manually
              run_query: ExtendedArgSpec(
                  [Parameter("sql", typ=TypedValue(str))],
                  name="run_query",
                  implementation=run_query_impl,
              )
          }
```


### Displaying and checking the type of an expression

You can use `pyanalyze.dump_value(expr)` to display the type pyanalyze infers for an expression. This can be
useful to understand errors or to debug why pyanalyze does not catch a particular issue. For example:

```python
from pyanalyze import dump_value

dump_value(1)  # value: KnownValue(val=1) (code: inference_failure)
```


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

Background
----------

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

Design
------

Fundamentally, the way pyanalyze works is that it tries to infer, with as much precision as possible, what Python value or what kind of Python value each node in a file's AST corresponds to, and then uses that information to flag code that does something undesirable. Mostly, that involves identifying code that will cause the Python interpreter to throw an error at runtime, for example because it accesses an attribute that doesn't exist or because it passes incorrect arguments to a function. As much as possible, the script tries to evaluate whether an operation is allowed by asking Python whether it is: for example, whether the arguments to a function call are correct is decided by creating a function with the same arguments as the called function, calling it with the same arguments as in the call, and checking whether the call throws an error.

This is done by recursively visiting the AST of the file and building up a context of information gathered from previously visited nodes. For example, the `visit_ClassDef` method visits the body of the class within a context that indicates that AST nodes are part of the class, which enables method definitions within the class to infer the type of their `self` arguments as being the class. In some cases, the visitor will traverse the AST twice: once to collect places where names are set, and once again to check that every place a name is accessed is valid. This is necessary because functions may use names that are only defined later in the file.

### Name resolution

The name resolution component of pyanalyze makes it possible to connect usage of a Python variable with the place where it is defined.

Pyanalyze uses the `StackedScopes` class to simulate Python scoping rules. This class contains a stack of nested scopes, implemented as dictionaries, that contain names defined in a particular Python scope (e.g., a function). When the script needs to determine what a particular name refers to, it iterates through the scopes, starting at the top of the scope stack, until it finds a scope dictionary that contains the name. This is similar to how name lookup is implemented in Python itself. When a name that is accessed in Python code is not found in any scope object, `test_scope.py` will throw an error with code `undefined_name`.

When the script is run on a file, the scopes object is initialized with two scope levels containing builtin objects such as `len` and `Exception` and the file's module-level globals (found by importing the file and inspecting its `__dict__`). When it inspects the AST, it adds names that it finds in assignment context into the appropriate nested scope. For example, when the scripts sees a `FunctionDef` AST node, it adds a new function-level scope, and if the function contains a statement like `x = 1`, it will add the variable `x` to the function's scope. Then when the function accesses the variable `x`, the script can retrieve it from the function-level scope in the `StackedScopes` object.

The following scope types exist:

-   `builtin_scope` is at the bottom of every scope stack and contains standard Python builtin objects.
-   `module_scope` is always right above builtin_scope and contains module-global names, such as classes and functions defined at the global level in the file.
-   `class_scope` is entered whenever the AST visitor encounters a class definition. It can contain nested class or function scopes.
-   `function_scope` is entered for each function definition.

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

-   `if x is (not) None`
-   `if (not) x`
-   `if isinstance(x, <some type>)`

Constraints are used to restrict the types of:

-   Local variables
-   Instance variables (e.g., after `if self.x is None`, the type of `self.x` is restricted)
-   Nonlocal variables (variables defined in enclosing scopes)

### Type and value inference

Just knowing that a name has been defined doesn't tell what you can do with the value stored for the name. To get this information, each node visit method in `test_scope.py` can return an instance of the `Value` class representing the Python value that corresponds to the AST node. We also use type annotations in the code under consideration to get types for more values. Scope dictionaries also store `Value` instances to represent the values associated with names.

The following subclasses of `Value` exist:

-   `UnresolvedValue` (with a single instance, `UNRESOLVED_VALUE`), representing that the script knows nothing about the value a node can contain. For example, if a file contains only the function `def f(x): return x`, the name `x` will have `UNRESOLVED_VALUE` as its value within the function, because there is no information to determine what value it can contain.
-   `KnownValue` represents a value for which the script knows the concrete Python value. If a file contains the line `x = 1` and no other assignments to `x`, `x` will contain `KnownValue(1)`.
-   `TypedValue` represents that the script knows the type but not the exact value. If the only assignment to `x` is a line `x = int(some_function())`, the script infers that `x` contains `TypedValue(int)`. More generally, the script infers any call to a class as resulting in an instance of that class. The type is also inferred for the `self` argument of methods, for comprehensions, for function arguments with type annotations, and in a few other cases. This class has several subtypes:
    -   `NewTypeValue` corresponds to [`typing.NewType`](https://docs.python.org/3/library/typing.html#newtype); it indicates a distinct type that is identical to some other type at runtime. At Quora we use newtypes for helper types like `qtype.Uid`.
    -   `GenericValue` corresponds to generics, like `List[int]`.
-   `MultiValuedValue` indicates that multiple values are possible, for example because a variable is assigned to in multiple places. After the line `x = 1 if condition() else 'foo'`, `x` will contain `MultiValuedValue([KnownValue(1), KnownValue('foo')])`. This corresponds to [`typing.Union`](https://docs.python.org/3/library/typing.html#typing.Union).
-   `UnboundMethodValue` indicates that the value is a method, but that we don't have the instance the method is bound to. This often comes up when a method in a class `SomeClass` contains code like `self.some_other_method`: we know that self is a `TypedValue(SomeClass)` and that `SomeClass` has a method `some_other_method`, but we don't have the instance that `self.some_other_method` will be bound to, so we can't resolve a `KnownValue` for it. Returning an `UnboundMethodValue` in this case makes it still possible to check whether the arguments to the method are valid.
-   `ReferencingValue` represents a value that is a reference to a name in some other scopes. This is used to implement the `global` statement: `global x` creates a `ReferencingValue` referencing the `x` variable in the module scope. Assignments to it will affect the referenced value.
-   `SubclassValue` represents a class object of a class or its subclass. For example, in a classmethod, the type of the `cls` argument is a `SubclassValue` of the class the classmethod is defined in. At runtime, it is either this class or a subclass.
-   `NoReturnValue` indicates that a function will never return (e.g., because it always throws an error), corresponding to [`typing.NoReturn`](https://docs.python.org/3/library/typing.html#typing.NoReturn).

Each `Value` object has a method `is_value_compatible` that checks whether types are correct. The call `X.is_value_compatible(Y)` essentially answers the question: if we expect a value `X`, is it legal to pass a value `Y` instead? For example, `TypedValue(int).is_value_compatible(KnownValue(1))` will return True, because `1` is a valid `int`, but `TypedValue(int).is_value_compatible(KnownValue("1"))` will return False, because `"1"` is not.

### Call compatibility

When the visitor encounters a `Call` node (representing a function call) and it can resolve the object being called, it will check that the object can in fact be called and that it accepts the arguments given to it. This checks only the number of arguments and the names of keyword arguments, not their types.

The first step in implementing this check is to retrieve the argument specification (argspec) for the callee. Although Python provides the `inspect.getargspec` function to do this, this function doesn't work on classes and its result needs post-processing to remove the `self` argument from calls to bound methods. To figure out what arguments classes take, the argspec of their `__init__` method is retrieved. It is not always possible to programmatically determine what arguments built-in or Cythonized functions accept, but pyanalyze can often figure this out with the new Python 3 `inspect.signature` API or by using [typeshed](http://github.com/python/typeshed), a repository of types for standard library modules.

Once we have the argspec, we can figure out whether the arguments passed to the callee in the AST node under consideration are compatible with the argspec. The semantics of Python calls are sufficiently complicated that it seemed simplest to generate code that contains a function with the argspec and a call to that function with the node's arguments, which can be `exec`'ed to determine whether the call is valid. All default values and all arguments to the call are set to `None`. In verbose mode, this generated code is printed out:

```
$ cat call_example.py
def function(foo, bar=3, baz='baz'):
    return str(foo * bar) + baz

if False:  # to make the module importable
    function(2, bar=2, bax='2')
$ python -m pyanalyze -vv call_example.py
Checking file: ('call_example.py', 3469)
Code to execute:
def str(self, *args, **kwargs):
    return __builtin__.locals()

Variables from function call: {'self': TypedValue(typ=<class 'str'>), 'args': (UnresolvedValue(),), 'kwargs': {}}
Code to execute:
def function(foo, bar=__default_bar, baz=__default_baz):
    return __builtin__.locals()


TypeError("function() got an unexpected keyword argument 'bax'") (code: incompatible_call)
In call_example.py at line 5:
   2:     return str(foo * bar) + baz
   3:
   4: if False:  # to make the module importable
   5:     function(2, bar=2, bax='2')
          ^
```

### Non-existent object attributes

Python throws a runtime `AttributeError` when you try to access an object attribute that doesn't exist. `test_scope.py` can statically find some kinds of code that will access non-existent attribute. The simpler case is when code accesses an attribute of a `KnownValue` , like in a file that has `import os` and then accesses `os.ptah`. In this case, we know the value that `os` contains, so we can try to access the attribute `ptah` on it, and show an error if the attribute lookup fails. Similarly, `os.path` will return a `KnownValue` of the `os.path` module, so that we can also check attribute lookups on `os.path`.

Another class of bugs involves objects accessing attributes on `self` that don't exist. For example, an object may set `self.promote` in its `__init__` method, but then access `self.promotion` in its `tree` method. To detect such cases, pyanalyze uses the `ClassAttributeChecker` class. This class keeps a record of every node where an attribute is written or read on a `TypedValue`. After checking all code that uses the class, it then takes the difference between the sets of read and written values and shows an error for every attribute that is read but never written. This approach is complicated by inheritance---subclasses may read values only written on the superclass, and vice versa. Therefore, the check doesn't trigger for any attribute that is set on any superclass or subclass of the class under consideration. It also doesn't trigger for any attributes of a class that has a base class that wasn't itself examined by the `ClassAttributeChecker`. This was needed to deal with Thrift classes that used attributes defined in superclasses outside of code checked by pyanalyze. Two superclasses are excluded from this, so that undefined attributes are flagged on their subclasses even though test_scope.py hasn't examined their definitions: `object` (the superclass of every class) and `qutils.webnode2.Component` (which doesn't define any attributes that are read by its subclasses).

### Finding unused code

Because pyanalyze tries to resolve all names and attribute lookups in code in a package, it was easy to extend it to determine which of the classes and functions defined in the package aren't accessed in any other code. This is done by recording every name and attribute lookup that results in a `KnownValue` containing a function or class defined in the package. After the AST visitor run, it compares the set of accessed objects with another set of all the functions and classes that are defined in submodules of the package. All objects that appear in the second set but not the first are probably unused. (There are false positives, such as functions that are registered in some registry by decorators, or those that are called from outside of `a` itself.) This check can be run by passing the `--find-unused` argument to pyanalyze.

Type system
-----------

Pyanalyze partially supports the Python type system, as specified in [PEP 484](https://www.python.org/dev/peps/pep-0484/) and in the [Python documentation](https://docs.python.org/3/library/typing.html). It uses type annotations to infer types and checks for type compatibility in calls and return types. Supported type system features include generics like `List[int]`, `NewType`, and `TypedDict`.

However, support for some features is still missing, including:

- Callable types
- Overloaded functions
- Type variables
- Protocols

Limitations
-----------

Python is sufficiently dynamic that almost any check like the ones run by pyanalyze will inevitably have false positives: cases where the script sees an error, but the code in fact runs fine. Attributes may be added at runtime in hard-to-detect ways, variables may be created by direct manipulation of the `globals()` dictionary, and the `mock` module can change anything into anything. Although pyanalyze has a number of whitelists to deal with these false positives, it is usually better to write code in a way that doesn't require use of the whitelist: code that's easier for the script to understand is probably also easier for humans to understand.

Just as the script inevitably has false positives, it equally inevitably cannot find all code that will throw a runtime error. It is generally impossible to statically determine what a program does or whether it runs successfully without actually running the program. Pyanalyze doesn't check program logic and it cannot always determine exactly what value a variable will have. It is no substitute for unit tests.

Developing pyanalyze
--------------------

Pyanalyze has hundreds of unit tests that check its behavior. To run them, you can just run `pytest` in the project directory.

The code is formatted using [Black](https://github.com/psf/black).
