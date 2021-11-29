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

## Developing pyanalyze

Pyanalyze has hundreds of unit tests that check its behavior. To run them, you can just run `pytest` in the project directory.

The code is formatted using [Black](https://github.com/psf/black).

## Documentation

Documentation is available at [ReadTheDocs](https://pyanalyze.readthedocs.io/en/latest/)
or on [GitHub](https://github.com/quora/pyanalyze/tree/master/docs).
