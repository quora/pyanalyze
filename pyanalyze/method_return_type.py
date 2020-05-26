from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

"""

test_scope module for checking the return type of overridden methods.

"""

import inspect

from .error_code import ErrorCode
from . import value

NoneType = type(None)


def check_no_return(node, visitor, method_name):
    """When a function returns nothing, checks that that is correct.

    Invoked when the method returns just 'return' or contains no return statement at all.

    """
    for expected_return_type in _get_expected_return_types(visitor, method_name):
        if not issubclass(NoneType, expected_return_type):
            visitor.show_error(
                node,
                "Method %s should return a %s but returns nothing"
                % (method_name, expected_return_type),
                error_code=ErrorCode.invalid_method_return_type,
            )


def check_return_value(node, visitor, return_value, method_name):
    """When a function returns a value, checks that that value is of the correct type."""
    for expected_return_type in _get_expected_return_types(visitor, method_name):
        if isinstance(return_value, (value.KnownValue, value.TypedValue)):
            if not return_value.is_type(expected_return_type):
                visitor.show_error(
                    node,
                    "Method %s should return a %s but returns %s instead"
                    % (method_name, expected_return_type, return_value),
                    error_code=ErrorCode.invalid_method_return_type,
                )
        elif isinstance(return_value, value.MultiValuedValue):
            if any(
                isinstance(val, (value.TypedValue, value.KnownValue))
                and not val.is_type(expected_return_type)
                for val in return_value.vals
            ):
                visitor.show_error(
                    node,
                    "Method %s should return a %s but returns %s instead"
                    % (method_name, expected_return_type, return_value),
                    error_code=ErrorCode.invalid_method_return_type,
                )


def _get_expected_return_types(visitor, method_name):
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
    for base_cls in inspect.getmro(cls):
        if base_cls in return_types and method_name in return_types[base_cls]:
            yield return_types[base_cls][method_name]
            return
