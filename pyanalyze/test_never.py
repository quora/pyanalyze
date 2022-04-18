# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before
from .value import NO_RETURN_VALUE, TypedValue


class TestAnnotations(TestNameCheckVisitorBase):
    @assert_passes()
    def test_te_never(self):
        from typing_extensions import Never

        def capybara(n: Never, o: "Never"):
            assert_is_value(n, NO_RETURN_VALUE)
            assert_is_value(o, NO_RETURN_VALUE)

    @skip_before((3, 11))
    @assert_passes()
    def test_never(self):
        from typing import Never

        def capybara(n: Never, o: "Never"):
            assert_is_value(n, NO_RETURN_VALUE)
            assert_is_value(o, NO_RETURN_VALUE)

    @assert_passes()
    def test_typing_noreturn(self):
        from typing import NoReturn

        def capybara(n: NoReturn, o: "NoReturn"):
            assert_is_value(n, NO_RETURN_VALUE)
            assert_is_value(o, NO_RETURN_VALUE)


class TestNoReturn(TestNameCheckVisitorBase):
    @assert_passes()
    def test_no_return(self):
        from typing import Optional

        from typing_extensions import NoReturn

        def f() -> NoReturn:
            raise Exception

        def capybara(x: Optional[int]) -> None:
            if x is None:
                f()
            assert_is_value(x, TypedValue(int))

    @assert_passes()
    def test_no_return_parameter(self):
        from typing_extensions import NoReturn

        def assert_unreachable(x: NoReturn) -> None:
            pass

        def capybara():
            assert_unreachable(1)  # E: incompatible_argument

    @assert_passes()
    def test_assignability(self):
        from typing_extensions import NoReturn

        def takes_never(x: NoReturn):
            print(x)


class TestAssertNever(TestNameCheckVisitorBase):
    @assert_passes()
    def test_if(self):
        from typing import Union

        from pyanalyze.tests import assert_never

        def capybara(x: Union[int, str]) -> None:
            if isinstance(x, int):
                print("int")
            elif isinstance(x, str):
                print("str")
            else:
                assert_never(x)


class TestNeverCall(TestNameCheckVisitorBase):
    @assert_passes()
    def test_empty_list(self):
        def callee(a: int):
            pass

        def capybara():
            for args in []:
                callee(*args)

            for kwargs in []:
                callee(**kwargs)
