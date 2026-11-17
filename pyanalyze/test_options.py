# static analysis: ignore
from .asynq_checker import NonAsynqModules
from .error_code import ErrorCode
from .options import ConfigOption, Options
from .signature import MaximumPositionalArgs


def _error_code_option(code: ErrorCode, enabled: bool, **kwargs) -> ConfigOption:
    return ConfigOption.registry[code.name](enabled, **kwargs)


def test_get_value_for_is_stable() -> None:
    options = Options.from_option_list([MaximumPositionalArgs(3)])
    assert options.get_value_for(MaximumPositionalArgs) == 3
    assert options.get_value_for(MaximumPositionalArgs) == 3

    defaults = Options.from_option_list()
    expected = MaximumPositionalArgs.default_value
    assert defaults.get_value_for(MaximumPositionalArgs) == expected
    assert defaults.get_value_for(MaximumPositionalArgs) == expected


def test_concatenated_option_returns_fresh_list() -> None:
    options = Options.from_option_list([NonAsynqModules(["capybara"])])

    first = options.get_value_for(NonAsynqModules)
    assert "capybara" in first
    expected = list(first)
    first.append("mutated")

    second = options.get_value_for(NonAsynqModules)
    assert second == expected
    assert second is not first


def test_for_module_does_not_share_values() -> None:
    options = Options.from_option_list(
        [
            MaximumPositionalArgs(3),
            MaximumPositionalArgs(5, applicable_to=("capybara",)),
            NonAsynqModules(["pacarana"], applicable_to=("capybara",)),
        ]
    )
    # Populate the cache of the parent before deriving module options from it.
    assert options.get_value_for(MaximumPositionalArgs) == 3
    assert "pacarana" not in options.get_value_for(NonAsynqModules)

    for_capybara = options.for_module(("capybara", "hutia"))
    assert for_capybara.get_value_for(MaximumPositionalArgs) == 5
    assert "pacarana" in for_capybara.get_value_for(NonAsynqModules)

    for_other = options.for_module(("paca",))
    assert for_other.get_value_for(MaximumPositionalArgs) == 3
    assert options.get_value_for(MaximumPositionalArgs) == 3


def test_is_error_code_enabled() -> None:
    code = ErrorCode.missing_return_annotation
    default = ConfigOption.registry[code.name].default_value
    assert Options.from_option_list().is_error_code_enabled(code) is default

    options = Options.from_option_list(
        [
            _error_code_option(code, not default),
            _error_code_option(code, default, applicable_to=("capybara",)),
        ]
    )
    assert options.is_error_code_enabled(code) is not default
    assert options.is_error_code_enabled(code) is not default
    assert options.for_module(("capybara",)).is_error_code_enabled(code) is default
