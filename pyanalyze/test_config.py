"""

Configuration file specific to tests.

"""
from typing import Dict, Optional

from .arg_spec import ArgSpecCache
from .signature import (
    CallContext,
    ConcreteSignature,
    OverloadedSignature,
    Signature,
    SigParameter,
)
from .config import Config
from . import tests
from . import value


def _failing_impl(ctx: CallContext) -> value.Value:
    ctx.show_error("Always errors")
    return value.AnyValue(value.AnySource.error)


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

    def get_constructor(self, cls: type) -> Optional[Signature]:
        """Return a constructor signature for this class.

        May return either a function that pyanalyze will use the signature of, an inspect
        Signature object, or a pyanalyze Signature object. The function or signature
        should take a self parameter.

        """
        if issubclass(cls, tests.KeywordOnlyArguments):
            return Signature.make(
                [
                    SigParameter("self", kind=SigParameter.POSITIONAL_ONLY),
                    SigParameter("args", kind=SigParameter.VAR_POSITIONAL),
                    SigParameter(
                        "kwonly_arg",
                        kind=SigParameter.KEYWORD_ONLY,
                        default=value.KnownValue(None),
                    ),
                ],
                callable=tests.KeywordOnlyArguments.__init__,
            )
        return None

    def get_known_argspecs(
        self, arg_spec_cache: ArgSpecCache
    ) -> Dict[object, ConcreteSignature]:
        failing_impl_sig = arg_spec_cache.get_argspec(
            tests.FailingImpl, impl=_failing_impl
        )
        assert isinstance(failing_impl_sig, Signature), failing_impl_sig
        return {
            tests.takes_kwonly_argument: Signature.make(
                [
                    SigParameter("a"),
                    SigParameter(
                        "kwonly_arg",
                        SigParameter.KEYWORD_ONLY,
                        annotation=value.TypedValue(bool),
                    ),
                ],
                callable=tests.takes_kwonly_argument,
            ),
            tests.FailingImpl: failing_impl_sig,
            tests.overloaded: OverloadedSignature(
                [
                    Signature.make(
                        [], value.TypedValue(int), callable=tests.overloaded
                    ),
                    Signature.make(
                        [
                            SigParameter(
                                "x",
                                SigParameter.POSITIONAL_ONLY,
                                annotation=value.TypedValue(str),
                            )
                        ],
                        value.TypedValue(str),
                        callable=tests.overloaded,
                    ),
                ]
            ),
        }

    def unwrap_cls(self, cls: type) -> type:
        """Does any application-specific unwrapping logic for wrapper classes."""
        if (
            isinstance(cls, type)
            and issubclass(cls, tests.Wrapper)
            and cls is not tests.Wrapper
        ):
            return cls.base
        return cls
