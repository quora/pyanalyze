# Design

Fundamentally, the way pyanalyze works is that it tries to infer, with as much precision as possible, what Python value or what kind of Python value each node in a file's AST corresponds to, and then uses that information to flag code that does something undesirable. Mostly, that involves identifying code that will cause the Python interpreter to throw an error at runtime, for example because it accesses an attribute that doesn't exist or because it passes incorrect arguments to a function.

This is done by recursively visiting the AST of the file and building up a context of information gathered from previously visited nodes. For example, the `visit_ClassDef` method visits the body of the class within a context that indicates that AST nodes are part of the class, which enables method definitions within the class to infer the type of their `self` arguments as being the class. In some cases, the visitor will traverse the AST twice: once to collect places where names are set, and once again to check that every place a name is accessed is valid. This is necessary because functions may use names that are only defined later in the file.

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

## Name resolution

The name resolution component of pyanalyze makes it possible to connect usage of a Python variable with the place where it is defined.

Pyanalyze uses the `StackedScopes` class to simulate Python scoping rules. This class contains a stack of nested scopes, implemented as dictionaries, that contain names defined in a particular Python scope (e.g., a function). When we need to determine what a particular name refers to, we iterate through the scopes, starting at the top of the scope stack, until we find a scope dictionary that contains the name. This is similar to how name lookup is implemented in Python itself. When a name that is accessed in Python code is not found in any scope object, pyanalyze will throw an error with code `undefined_name`.

When pyanalyze is run on a file, the scopes object is initialized with two scope levels containing builtin objects such as `len` and `Exception` and the file's module-level globals (found by importing the file and inspecting its `__dict__`). When it inspects the AST, it adds names that it finds in assignment context into the appropriate nested scope. For example, when it sees a `FunctionDef` AST node, it adds a new function-level scope, and if the function contains a statement like `x = 1`, it will add the variable `x` to the function's scope. Then when the function accesses the variable `x`, it can retrieve it from the function-level scope in the `StackedScopes` object.

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
- `if x == <some value>`
- `if x in [<some>, <values>]`

Constraints are used to restrict the types of:

- Local variables
- Instance variables (e.g., after `if self.x is None`, the type of `self.x` is restricted)
- Nonlocal variables (variables defined in enclosing scopes)

## Type and value inference

Just knowing that a name has been defined doesn't tell what you can do with the value stored for the name. To get this information, each node visit method in `test_scope.py` can return an instance of the `Value` class representing the Python value that corresponds to the AST node. We also use type annotations in the code under consideration to get types for more values. Scope dictionaries also store `Value` instances to represent the values associated with names.

The following subclasses of `Value` exist:

- `AnyValue`, representing that we know nothing about the value a node can contain. For example, if a file contains only the function `def f(x): return x`, the name `x` will have an `AnyValue` as its value within the function, because there is no information to determine what value it can contain.
- `KnownValue` represents a value for which we know the concrete Python value. If a file contains the line `x = 1` and no other assignments to `x`, `x` will contain `KnownValue(1)`.
- `TypedValue` represents that we know the type but not the exact value. If the only assignment to `x` is a line `x = int(some_function())`, we infer that `x` contains `TypedValue(int)`. More generally, we infer any call to a class as resulting in an instance of that class. The type is also inferred for the `self` argument of methods, for comprehensions, for function arguments with type annotations, and in a few other cases. This class has several subtypes:
  - `NewTypeValue` corresponds to [`typing.NewType`](https://docs.python.org/3/library/typing.html#newtype); it indicates a distinct type that is identical to some other type at runtime. At Quora we use newtypes for helper types like `qtype.Uid`.
  - `GenericValue` corresponds to generics, like `List[int]`.
- `MultiValuedValue` indicates that multiple values are possible, for example because a variable is assigned to in multiple places. After the line `x = 1 if condition() else 'foo'`, `x` will contain `MultiValuedValue([KnownValue(1), KnownValue('foo')])`. This corresponds to [`typing.Union`](https://docs.python.org/3/library/typing.html#typing.Union).
- `UnboundMethodValue` indicates that the value is a method, but that we don't have the instance the method is bound to. This often comes up when a method in a class `SomeClass` contains code like `self.some_other_method`: we know that self is a `TypedValue(SomeClass)` and that `SomeClass` has a method `some_other_method`, but we don't have the instance that `self.some_other_method` will be bound to, so we can't resolve a `KnownValue` for it. Returning an `UnboundMethodValue` in this case makes it still possible to check whether the arguments to the method are valid.
- `ReferencingValue` represents a value that is a reference to a name in some other scopes. This is used to implement the `global` statement: `global x` creates a `ReferencingValue` referencing the `x` variable in the module scope. Assignments to it will affect the referenced value.
- `SubclassValue` represents a class object of a class or its subclass. For example, in a classmethod, the type of the `cls` argument is a `SubclassValue` of the class the classmethod is defined in. At runtime, it is either this class or a subclass.
- `NoReturnValue` indicates that a function will never return (e.g., because it always throws an error), corresponding to [`typing.NoReturn`](https://docs.python.org/3/library/typing.html#typing.NoReturn).

Each `Value` object has a method `can_assign` that checks whether types are correct. The call `X.can_assign(Y, ctx)` essentially answers the question: if we expect a value `X`, is it legal to pass a value `Y` instead? For example, `TypedValue(int).can_assign(KnownValue(1), ctx)` will succeed, because `1` is a valid `int`, but `TypedValue(int).can_assign(KnownValue("1"), ctx)` will fail, because `"1"` is not. In order to help with type checking generics, the return value of `can_assign` is a (possibly empty) dictionary of type variables mapping to their values.

## Call compatibility

When the visitor encounters a `Call` node (representing a function call) and it can resolve the object being called, it will check that the object can in fact be called and that it accepts the arguments given to it. This checks only the number of arguments and the names of keyword arguments, not their types.

The first step in implementing this check is to retrieve the signature for the callee. Python provides the `inspect.signature` function to do this, but for some callables additional logic is required. In addition, pyanalyze uses [typeshed](http://github.com/python/typeshed), a repository of types for standard library modules, to produce more precise signatures.

Once we have the signature, we can figure out whether the arguments passed to the callee in the AST node under consideration are compatible with the signature. This is done by matching up the arguments passed by the call to the formal parameters in the signature. The abstraction also supports providing an _implementation function_ for a callable, a function that gets called with the types of the arguments to the function and that computes a more specific return type or checks the arguments.

## Non-existent object attributes

Python throws a runtime `AttributeError` when you try to access an object attribute that doesn't exist. Pyanalyze can statically find some kinds of code that will access non-existent attributes. The simpler case is when code accesses an attribute of a `KnownValue` , like in a file that has `import os` and then accesses `os.ptah`. In this case, we know the value that `os` contains, so we can try to access the attribute `ptah` on it, and show an error if the attribute lookup fails. Similarly, `os.path` will return a `KnownValue` of the `os.path` module, so that we can also check attribute lookups on `os.path`.

Another class of bugs involves objects accessing attributes on `self` that don't exist. For example, an object may set `self.promote` in its `__init__` method, but then access `self.promotion` in its `tree` method. To detect such cases, pyanalyze uses the `ClassAttributeChecker` class. This class keeps a record of every node where an attribute is written or read on a `TypedValue`. After checking all code that uses the class, it then takes the difference between the sets of read and written values and shows an error for every attribute that is read but never written. This approach is complicated by inheritance---subclasses may read values only written on the superclass, and vice versa. Therefore, the check doesn't trigger for any attribute that is set on any superclass or subclass of the class under consideration. It also doesn't trigger for any attributes of a class that has a base class that wasn't itself examined by the `ClassAttributeChecker`. This was needed to deal with Thrift classes that used attributes defined in superclasses outside of code checked by pyanalyze. Two superclasses are excluded from this, so that undefined attributes are flagged on their subclasses even though test_scope.py hasn't examined their definitions: `object` (the superclass of every class) and `qutils.webnode2.Component` (which doesn't define any attributes that are read by its subclasses).

## Finding unused code

Because pyanalyze tries to resolve all names and attribute lookups in code in a package, it was easy to extend it to determine which of the classes and functions defined in the package aren't accessed in any other code. This is done by recording every name and attribute lookup that results in a `KnownValue` containing a function or class defined in the package. After the AST visitor run, it compares the set of accessed objects with another set of all the functions and classes that are defined in submodules of the package. All objects that appear in the second set but not the first are probably unused. (There are false positives, such as functions that are registered in some registry by decorators, or those that are called from outside of `a` itself.) This check can be run by passing the `--find-unused` argument to pyanalyze.
