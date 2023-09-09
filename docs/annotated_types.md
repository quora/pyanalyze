# Support for `annotated_types`

Pyanalyze supports the [annotated-types](https://pypi.org/project/annotated-types/) library, which provides a set of common primitives to use in `Annotated` metadata.

For the `MultipleOf` check, pyanalyze follows Python semantics: values
are accepted if `value % multiple_of == 0`.

For the `Timezone` check, support for requiring string-based timezones is not implemented.
