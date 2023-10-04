# Support for `annotated_types`

Pyanalyze supports the [annotated-types](https://pypi.org/project/annotated-types/) library, which provides a set of common primitives to use in `Annotated` metadata.

This is useful for restricting the value of an object:

```python
from typing_extensions import Annotated
from annotated_types import Gt

def takes_gt_5(x: Annotated[int, Gt(5)]) -> None:
    assert x > 5, "number too small"

def caller() -> None:
    takes_gt_5(6)  # ok
    takes_gt_5(5)  # type checker error
```

Pyanalyze enforces these annotations strictly: if it cannot determine whether or
not a value fulfills the predicate, it shows an error. For example, the following
will be rejected:

```python
def caller(i: int) -> None:
    takes_gt_5(i)  # type checker error, as it may be less than 5
```

## Notes on specific predicates

Pyanalyze infers the interval attributes `Gt`, `Ge`, `Lt`, and `Le` based
on comparisons with literals:

```python
def caller(i: int) -> None:
    takes_gt_5(i)  # error

    if i > 5:
        takes_gt_5(i)  # accepted
```

Similarly, pyanalyze infers the `MinLen` and `MaxLen` attributes after checks
on `len()`.

For the `MultipleOf` check, pyanalyze follows Python semantics: values
are accepted if `value % multiple_of == 0`.

For the `Timezone` check, support for requiring string-based timezones is not implemented.
