"""

Error codes used by test_scope.

"""

import enum
from aenum import extend_enum

from .find_unused import used


class ErrorCode(enum.Enum):
    # internal
    bad_star_import = 1
    cant_import = 2
    unexpected_node = 3

    # undefined names and attributes
    undefined_name = 4
    undefined_attribute = 5
    attribute_is_never_set = 6

    # bad dict and set literals
    duplicate_dict_key = 7
    unhashable_key = 8

    # unsupported operations
    bad_unpack = 9
    unsupported_operation = 10

    # function calls and arguments
    not_callable = 11
    incompatible_call = 12
    method_first_arg = 13
    bad_super_call = 14

    # async
    impure_async_call = 15
    unnecessary_yield = 16

    class_variable_redefinition = 21
    bad_global = 22
    condition_always_true = 23  # deprecated
    inference_failure = 24
    bad_format_string = 25
    yield_without_value = 28
    invalid_method_return_type = 30
    missing_asynq = 31
    bad_exception = 32
    bad_async_yield = 34
    add_import = 35
    duplicate_yield = 36
    yield_in_comprehension = 37
    use_floor_div = 38
    task_needs_yield = 39
    mixing_bytes_and_text = 40  # deprecated
    bad_except_handler = 41
    implicit_non_ascii_string = 42
    missing_await = 43
    unused_variable = 44
    bad_nonlocal = 45
    non_boolean_in_boolean_context = 46  # deprecated
    use_fstrings = 47
    import_failed = 48
    unused_ignore = 49
    possibly_undefined_name = 50
    missing_f = 51
    incompatible_return_value = 52
    incompatible_argument = 53
    incompatible_default = 54
    internal_error = 55
    bad_yield_from = 56
    incompatible_assignment = 57
    invalid_typeddict_key = 58
    invalid_annotation = 59
    bare_ignore = 60
    duplicate_enum_member = 61
    missing_return_annotation = 62
    missing_parameter_annotation = 63
    type_always_true = 64
    value_always_true = 65
    type_does_not_support_bool = 66
    missing_return = 67
    no_return_may_return = 68
    implicit_reexport = 69


# Allow testing unannotated functions without too much fuss
DISABLED_IN_TESTS = {
    ErrorCode.missing_return_annotation,
    ErrorCode.missing_parameter_annotation,
}


DISABLED_BY_DEFAULT = {
    *DISABLED_IN_TESTS,
    ErrorCode.method_first_arg,
    ErrorCode.value_always_true,
    # TODO(jelle): This needs more work
    ErrorCode.unused_variable,
    ErrorCode.use_fstrings,
    ErrorCode.unused_ignore,
    ErrorCode.possibly_undefined_name,
    ErrorCode.missing_f,
    ErrorCode.bare_ignore,
    # TODO: turn this on
    ErrorCode.implicit_reexport,
}

ERROR_DESCRIPTION = {
    ErrorCode.bad_star_import: '"from ... import *" within a function.',
    ErrorCode.cant_import: "Internal error while checking a star import.",
    ErrorCode.unexpected_node: (
        "The script encountered a kind of code it does not know about."
    ),
    ErrorCode.undefined_name: "Usage of a variable that is never assigned to.",
    ErrorCode.undefined_attribute: (
        "Usage of an attribute (e.g. a function in a module) that does not exist."
    ),
    ErrorCode.attribute_is_never_set: (
        "An attribute that is read on objects of a particular type is never set on that"
        " object."
    ),
    ErrorCode.duplicate_dict_key: "Duplicate key in a dictionary.",
    ErrorCode.unhashable_key: "Key cannot be inserted into a set or dictionary.",
    ErrorCode.bad_unpack: "Error in an unpacking assignment.",
    ErrorCode.unsupported_operation: (
        "Usage of an operation such as subscripting on an object that does not"
        " support it."
    ),
    ErrorCode.not_callable: "Attempt to call an object that is not callable.",
    ErrorCode.incompatible_call: "Incompatible arguments to a function call.",
    ErrorCode.method_first_arg: "First argument to a method is not cls or self.",
    ErrorCode.bad_super_call: "Call to super() with invalid arguments.",
    ErrorCode.impure_async_call: (
        "Non-async call to an async function within another async function."
    ),
    ErrorCode.unnecessary_yield: "Unnecessary yield in async function.",
    ErrorCode.class_variable_redefinition: (
        "Redefinition of a class-level variable. Usually this means a duplicate method"
        " or enum value."
    ),
    ErrorCode.bad_global: "Bad global declaration.",
    ErrorCode.condition_always_true: "Condition is always true.",
    ErrorCode.inference_failure: "Internal error in value inference.",
    ErrorCode.bad_format_string: "Incorrect arguments to a %-formatted string.",
    ErrorCode.yield_without_value: "yield without a value in an async function",
    ErrorCode.invalid_method_return_type: (
        "An overridden method returns an object of the wrong type"
    ),
    ErrorCode.missing_asynq: "This function should have an @asynq() decorator",
    ErrorCode.bad_exception: "An object that is not an exception is raised",
    ErrorCode.bad_async_yield: "Yield of an invalid value in an async function",
    ErrorCode.add_import: "You should add an import",
    ErrorCode.duplicate_yield: "Duplicate yield of the same value in an async function",
    ErrorCode.yield_in_comprehension: "Yield within a comprehension",
    ErrorCode.use_floor_div: "Use // to divide two integers",
    ErrorCode.task_needs_yield: "You probably forgot to yield an async task",
    ErrorCode.mixing_bytes_and_text: "Do not mix str and unicode",
    ErrorCode.bad_except_handler: "Invalid except clause",
    ErrorCode.implicit_non_ascii_string: (
        "Non-ASCII bytestring without an explicit prefix"
    ),
    ErrorCode.missing_await: "Missing await in async code",
    ErrorCode.unused_variable: "Variable is not read after being written to",
    ErrorCode.bad_nonlocal: "Incorrect usage of nonlocal",
    ErrorCode.non_boolean_in_boolean_context: (
        "Object will always evaluate to True in boolean context"
    ),
    ErrorCode.use_fstrings: "Use f-strings instead of % formatting",
    ErrorCode.import_failed: "Failed to import module",
    ErrorCode.unused_ignore: "Unused '# static analysis: ignore' comment",
    ErrorCode.possibly_undefined_name: "Variable may be uninitialized",
    ErrorCode.missing_f: "f missing from f-string",
    ErrorCode.incompatible_return_value: "Incompatible return value",
    ErrorCode.incompatible_argument: "Incompatible argument type",
    ErrorCode.incompatible_default: "Default value incompatible with argument type",
    ErrorCode.internal_error: "Internal error; please report this as a bug",
    ErrorCode.bad_yield_from: "Incompatible type in yield from",
    ErrorCode.incompatible_assignment: "Incompatible variable assignment",
    ErrorCode.invalid_typeddict_key: "Invalid key in TypedDict",
    ErrorCode.invalid_annotation: "Invalid type annotation",
    ErrorCode.bare_ignore: "Ignore comment without an error code",
    ErrorCode.duplicate_enum_member: "Duplicate enum member",
    ErrorCode.missing_return_annotation: "Missing function return annotation",
    ErrorCode.missing_parameter_annotation: "Missing function parameter annotation",
    ErrorCode.type_always_true: "Type will always evaluate to 'True'",
    ErrorCode.value_always_true: "Value will always evaluate to 'True'",
    ErrorCode.type_does_not_support_bool: "Type does not support bool()",
    ErrorCode.missing_return: "Function may exit without returning a value",
    ErrorCode.no_return_may_return: "Function is annotated as NoReturn but may return",
    ErrorCode.implicit_reexport: "Use of implicitly re-exported name",
}


@used  # exposed as an API
def register_error_code(name: str, description: str) -> ErrorCode:
    """Register an additional error code. For use in extensions."""
    value = max(member.value for member in ErrorCode) + 1
    extend_enum(ErrorCode, name, value)
    member = ErrorCode[name]
    ERROR_DESCRIPTION[member] = description
    return member
