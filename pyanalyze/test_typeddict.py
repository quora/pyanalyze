# static analysis: ignore
from .implementation import assert_is_value
from .value import TypedDictValue, TypedValue
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestTypedDict(TestNameCheckVisitorBase):
    @assert_passes()
    def test_constructor(self):
        from typing_extensions import TypedDict, NotRequired

        class Capybara(TypedDict):
            x: int
            y: str

        class MaybeCapybara(TypedDict):
            x: int
            y: NotRequired[str]

        def capybara():
            cap = Capybara(x=1, y="2")
            assert_is_value(
                cap,
                TypedDictValue(
                    {"x": (True, TypedValue(int)), "y": (True, TypedValue(str))}
                ),
            )
            Capybara(x=1)  # E: incompatible_call

            maybe_cap = MaybeCapybara(x=1)
            assert_is_value(
                maybe_cap,
                TypedDictValue(
                    {"x": (True, TypedValue(int)), "y": (False, TypedValue(str))}
                ),
            )
