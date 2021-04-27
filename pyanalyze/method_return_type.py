"""

test_scope module for checking the return type of overridden methods.

"""

import ast
import inspect
from typing import Iterable, TYPE_CHECKING

from .error_code import ErrorCode
from . import type_object
from .value import Value, KnownValue, TypedValue, MultiValuedValue

if TYPE_CHECKING:
    from .name_check_visitor import NameCheckVisitor

NoneType = type(None)


def check_no_return(
    node: ast.AST, visitor: "NameCheckVisitor", method_name: str
) -> None:
    """When a function returns nothing, checks that that is correct.

    Invoked when the method returns just 'return' or contains no return statement at all.

    """
    for expected_return_type in _get_expected_return_types(visitor, method_name):
        if not issubclass(NoneType, expected_return_type):
            visitor.show_error(
                node,
                f"Method {method_name} should return a {expected_return_type} but"
                " returns nothing",
                error_code=ErrorCode.invalid_method_return_type,
            )


def check_return_value(
    node: ast.AST, visitor: "NameCheckVisitor", return_value: Value, method_name: str
) -> None:
    """When a function returns a value, checks that that value is of the correct type."""
    for expected_return_type in _get_expected_return_types(visitor, method_name):
        if isinstance(return_value, (KnownValue, TypedValue)):
            if not return_value.is_type(expected_return_type):
                visitor.show_error(
                    node,
                    f"Method {method_name} should return a {expected_return_type} but"
                    f" returns {return_value} instead",
                    error_code=ErrorCode.invalid_method_return_type,
                )
        elif isinstance(return_value, MultiValuedValue):
            if any(
                isinstance(val, (TypedValue, KnownValue))
                and not val.is_type(expected_return_type)
                for val in return_value.vals
            ):
                visitor.show_error(
                    node,
                    f"Method {method_name} should return a {expected_return_type} but"
                    f" returns {return_value} instead",
                    error_code=ErrorCode.invalid_method_return_type,
                )


def _get_expected_return_types(
    visitor: "NameCheckVisitor", method_name: str
) -> Iterable[type]:
    """Returns all expected return types that we should check for this method.

    Currently at most one expected return type will be returned.

    """
    if method_name is None:
        return

    cls = visitor.current_class
    if cls is None or not inspect.isclass(cls):
        # not a method
        return

    # Iterate over the MRO so that we can support specifying a different return type for a child
    # class
    return_types = visitor.config.METHOD_RETURN_TYPES
    for base_cls in type_object.get_mro(cls):
        if base_cls in return_types and method_name in return_types[base_cls]:
            yield return_types[base_cls][method_name]
            return
