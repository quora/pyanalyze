from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

"""

Configuration file specific to tests.

"""
from .arg_spec import ExtendedArgSpec, Parameter
from .config import Config
from . import tests
from . import value


class TestConfig(Config):
    """Configuration to run in test_scope's unit tests."""

    BASE_CLASSES_CHECKED_FOR_ASYNQ = {tests.CheckedForAsynq}
    METHODS_NOT_CHECKED_FOR_ASYNQ = {"not_checked"}
    METHOD_RETURN_TYPES = {
        tests.FixedMethodReturnType: {
            "should_return_none": type(None),
            "should_return_list": list,
        }
    }
    PROPERTIES_OF_KNOWN_TYPE = {
        tests.PropertyObject.string_property: value.TypedValue(str)
    }
    NAMES_OF_KNOWN_TYPE = {"proper_capybara": tests.PropertyObject}

    VARIABLE_NAME_VALUES = [
        value.VariableNameValue(["uid"]),
        value.VariableNameValue(["qid"]),
    ]

    CLASS_TO_KEYWORD_ONLY_ARGUMENTS = {tests.KeywordOnlyArguments: ["kwonly_arg"]}

    def get_known_argspecs(self, arg_spec_cache):
        return {
            tests.takes_kwonly_argument: ExtendedArgSpec(
                [Parameter("a")],
                name="takes_kwonly_argument",
                kwonly_args=[Parameter("kwonly_arg", typ=bool)],
            )
        }

    def unwrap_cls(self, cls):
        """Does any application-specific unwrapping logic for wrapper classes."""
        if (
            isinstance(cls, type)
            and issubclass(cls, tests.Wrapper)
            and cls is not tests.Wrapper
        ):
            return cls.base
        return cls
