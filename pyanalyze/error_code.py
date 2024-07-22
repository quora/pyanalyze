"""

Error codes used by test_scope.

"""

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator

import pyanalyze

from .find_unused import used


@dataclass(frozen=True)
class Error:
    __slots__ = ("name", "description")
    name: str
    description: str


class ErrorRegistry:
    errors: Dict[str, Error]

    def __init__(self, errors: Iterable[Error]) -> None:
        self.errors = {}
        for error in errors:
            self.errors[error.name] = error

    def register(self, name: str, description: str) -> Error:
        error = Error(name, description)
        self.errors[name] = error
        return error

    def __getattr__(self, name: str) -> Error:
        try:
            return self.errors[name]
        except KeyError:
            raise AttributeError(name) from None

    def __iter__(self) -> Iterator[Error]:
        return iter(self.errors.values())


ErrorCode = ErrorRegistry(
    [
        Error("bad_star_import", '"from ... import *" within a function.'),
        Error("cant_import", "Internal error while checking a star import."),
        Error(
            "unexpected_node",
            "The script encountered a kind of code it does not know about.",
        ),
        Error("undefined_name", "Usage of a variable that is never assigned to."),
        Error(
            "undefined_attribute",
            "Usage of an attribute (e.g. a function in a module) that does not exist.",
        ),
        Error(
            "attribute_is_never_set",
            "An attribute that is read on objects of a particular type is"
            " never set on that object.",
        ),
        Error("duplicate_dict_key", "Duplicate key in a dictionary."),
        Error("unhashable_key", "Key cannot be inserted into a set or dictionary."),
        Error("bad_unpack", "Error in an unpacking assignment."),
        Error(
            "unsupported_operation",
            "Usage of an operation such as subscripting on an object that does not support it.",
        ),
        Error("not_callable", "Attempt to call an object that is not callable."),
        Error("incompatible_call", "Incompatible arguments to a function call."),
        Error("method_first_arg", "First argument to a method is not cls or self."),
        Error("bad_super_call", "Call to super() with invalid arguments."),
        Error(
            "impure_async_call",
            "Non-async call to an async function within another async function.",
        ),
        Error("unnecessary_yield", "Unnecessary yield in async function."),
        Error(
            "class_variable_redefinition",
            "Redefinition of a class-level variable."
            " Usually this means a duplicate method or enum value.",
        ),
        Error("bad_global", "Bad global declaration."),
        Error("condition_always_true", "Condition is always true."),
        Error("inference_failure", "Incorrectly inferred type."),
        Error("reveal_type", "Revealed type."),
        Error("bad_format_string", "Incorrect arguments to a %-formatted string."),
        Error("yield_without_value", "yield without a value in an async function"),
        Error(
            "invalid_method_return_type",
            "An overridden method returns an object of the wrong type",
        ),
        Error("missing_asynq", "This function should have an @asynq() decorator"),
        Error("bad_exception", "An object that is not an exception is raised"),
        Error("bad_async_yield", "Yield of an invalid value in an async function"),
        Error("add_import", "You should add an import"),
        Error(
            "duplicate_yield", "Duplicate yield of the same value in an async function"
        ),
        Error("yield_in_comprehension", "Yield within a comprehension"),
        Error("use_floor_div", "Use // to divide two integers"),
        Error("task_needs_yield", "You probably forgot to yield an async task"),
        Error("mixing_bytes_and_text", "Do not mix str and unicode"),
        Error("bad_except_handler", "Invalid except clause"),
        Error(
            "implicit_non_ascii_string",
            "Non-ASCII bytestring without an explicit prefix",
        ),
        Error("missing_await", "Missing await in async code"),
        Error("unused_variable", "Variable is not read after being written to"),
        Error("bad_nonlocal", "Incorrect usage of nonlocal"),
        Error(
            "non_boolean_in_boolean_context",
            "Object will always evaluate to True in boolean context",
        ),
        Error("use_fstrings", "Use f-strings instead of % formatting"),
        Error("import_failed", "Failed to import module"),
        Error("unused_ignore", "Unused '# static analysis: ignore' comment"),
        Error("possibly_undefined_name", "Variable may be uninitialized"),
        Error("missing_f", "f missing from f-string"),
        Error("incompatible_return_value", "Incompatible return value"),
        Error("incompatible_argument", "Incompatible argument type"),
        Error("incompatible_default", "Default value incompatible with argument type"),
        Error("internal_error", "Internal error; please report this as a bug"),
        Error("bad_yield_from", "Incompatible type in yield from"),
        Error("incompatible_assignment", "Incompatible variable assignment"),
        Error("invalid_typeddict_key", "Invalid key in TypedDict"),
        Error("invalid_annotation", "Invalid type annotation"),
        Error("bare_ignore", "Ignore comment without an error code"),
        Error("duplicate_enum_member", "Duplicate enum member"),
        Error("missing_return_annotation", "Missing function return annotation"),
        Error("missing_parameter_annotation", "Missing function parameter annotation"),
        Error("type_always_true", "Type will always evaluate to 'True'"),
        Error("value_always_true", "Value will always evaluate to 'True'"),
        Error("type_does_not_support_bool", "Type does not support bool()"),
        Error("missing_return", "Function may exit without returning a value"),
        Error(
            "no_return_may_return", "Function is annotated as NoReturn but may return"
        ),
        Error("implicit_reexport", "Use of implicitly re-exported name"),
        Error("invalid_context_manager", "Use of invalid object in with or async with"),
        Error("suggested_return_type", "Suggested return type"),
        Error("suggested_parameter_type", "Suggested parameter type"),
        Error("incompatible_override", "Class attribute incompatible with base class"),
        Error("impossible_pattern", "Pattern can never match"),
        Error("bad_match", "Invalid type in match statement"),
        Error("bad_evaluator", "Invalid code in type evaluator"),
        Error("implicit_any", "Value is inferred as Any"),
        Error("already_declared", "Name is already declared"),
        Error("invalid_annotated_assignment", "Invalid annotated assignment"),
        Error("unused_assignment", "Assigned value is never used"),
        Error("incompatible_yield", "Incompatible yield type"),
        Error("deprecated", "Use of deprecated feature"),
        Error("invalid_import", "Invalid import"),
        Error(
            "too_many_positional_args",
            "Call with many positional arguments should use keyword arguments",
        ),
        Error("invalid_override_decorator", "@override decorator in invalid location"),
        Error("override_does_not_override", "Method does not override any base method"),
        Error("missing_generic_parameters", "Missing type parameters for generic type"),
        Error("disallowed_import", "Disallowed import"),
        Error(
            "typeis_must_be_subtype",
            "TypeIs narrowed type must be a subtype of the input type",
        ),
        Error("invalid_typeguard", "Invalid use of TypeGuard or TypeIs"),
        Error("readonly_typeddict", "TypedDict is read-only"),
        Error("generator_return", "Generator must return an iterable"),
        Error("unsafe_comparison", "Non-overlapping equality checks"),
        Error("must_use", "Value cannot be discarded"),
    ]
)


# Allow testing unannotated functions without too much fuss
DISABLED_IN_TESTS = {
    ErrorCode.missing_return_annotation,
    ErrorCode.missing_parameter_annotation,
    ErrorCode.suggested_return_type,
    ErrorCode.suggested_parameter_type,
    ErrorCode.implicit_any,
}


DISABLED_BY_DEFAULT = {
    *DISABLED_IN_TESTS,
    ErrorCode.method_first_arg,
    ErrorCode.value_always_true,
    ErrorCode.use_fstrings,
    ErrorCode.unused_ignore,
    ErrorCode.missing_f,
    ErrorCode.bare_ignore,
    ErrorCode.too_many_positional_args,
    # TODO: turn this on
    ErrorCode.implicit_reexport,
    ErrorCode.incompatible_override,
    ErrorCode.missing_generic_parameters,
}


@used  # exposed as an API
def register_error_code(name: str, description: str) -> Error:
    """Register an additional error code. For use in extensions."""
    member = ErrorCode.register(name, description)
    type(
        name,
        (pyanalyze.options.BooleanOption,),
        {
            "__doc__": description,
            "name": name,
            "default_value": True,
            "should_create_command_line_option": False,
        },
    )
    return member
