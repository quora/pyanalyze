"""

Configuration file specific to tests.

"""
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import tests, value
from .arg_spec import ArgSpecCache
from .error_code import ErrorCode, register_error_code
from .find_unused import used

from .options import Options
from .signature import (
    CallContext,
    ConcreteSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
    SigParameter,
)

register_error_code("internal_test", "Used in an internal test")


def _failing_impl(ctx: CallContext) -> value.Value:
    ctx.show_error("Always errors")
    return value.AnyValue(value.AnySource.error)


def _custom_code_impl(ctx: CallContext) -> value.Value:
    ctx.show_error("Always errors", ErrorCode.internal_test)
    return value.AnyValue(value.AnySource.error)


@used  # in test.toml
def get_constructor(cls: type) -> Optional[Signature]:
    """Return a constructor signature for this class.

    May return either a function that pyanalyze will use the signature of, an inspect
    Signature object, or a pyanalyze Signature object. The function or signature
    should take a self parameter.

    """
    if issubclass(cls, tests.KeywordOnlyArguments):
        return Signature.make(
            [
                SigParameter("self", kind=ParameterKind.POSITIONAL_ONLY),
                SigParameter("args", kind=ParameterKind.VAR_POSITIONAL),
                SigParameter(
                    "kwonly_arg",
                    kind=ParameterKind.KEYWORD_ONLY,
                    default=value.KnownValue(None),
                ),
            ],
            callable=tests.KeywordOnlyArguments.__init__,
        )
    return None


@used  # in test.toml
def get_known_signatures(
    arg_spec_cache: ArgSpecCache,
) -> Dict[object, ConcreteSignature]:
    failing_impl_sig = arg_spec_cache.get_argspec(tests.FailingImpl, impl=_failing_impl)
    custom_sig = arg_spec_cache.get_argspec(tests.custom_code, impl=_custom_code_impl)
    assert isinstance(custom_sig, Signature), custom_sig
    assert isinstance(failing_impl_sig, Signature), failing_impl_sig
    return {
        tests.takes_kwonly_argument: Signature.make(
            [
                SigParameter("a"),
                SigParameter(
                    "kwonly_arg",
                    ParameterKind.KEYWORD_ONLY,
                    annotation=value.TypedValue(bool),
                ),
            ],
            callable=tests.takes_kwonly_argument,
        ),
        tests.FailingImpl: failing_impl_sig,
        tests.custom_code: custom_sig,
        tests.overloaded: OverloadedSignature(
            [
                Signature.make([], value.TypedValue(int), callable=tests.overloaded),
                Signature.make(
                    [
                        SigParameter(
                            "x",
                            ParameterKind.POSITIONAL_ONLY,
                            annotation=value.TypedValue(str),
                        )
                    ],
                    value.TypedValue(str),
                    callable=tests.overloaded,
                ),
            ]
        ),
    }


@used  # in test.toml
def unwrap_class(cls: type) -> type:
    """Does any application-specific unwrapping logic for wrapper classes."""
    if (
        isinstance(cls, type)
        and issubclass(cls, tests.Wrapper)
        and cls is not tests.Wrapper
    ):
        return cls.base
    return cls


class StringField:
    pass


@used  # in test.toml
def transform_class_attribute(
    attr: object,
) -> Optional[Tuple[value.Value, value.Value]]:
    """Transforms a StringField attribute."""
    if isinstance(attr, StringField):
        return value.TypedValue(str), value.NO_RETURN_VALUE
    return None


CONFIG_PATH = Path(__file__).parent / "test.toml"
TEST_OPTIONS = Options.from_option_list(config_file_path=CONFIG_PATH)
