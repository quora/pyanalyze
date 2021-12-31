# static analysis: ignore
from .value import TypedValue, assert_is_value
from .test_node_visitor import assert_passes
from .test_name_check_visitor import TestNameCheckVisitorBase
from .extensions import is_keyword, is_positional, is_provided, is_of_type, show_error


class TestTypeEvaluation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_is_provided(self):
        from pyanalyze.extensions import evaluated
        from typing import Union

        @evaluated
        def simple_evaluated(x: int, y: str = ""):
            if is_provided(y):
                return int
            else:
                return str

        def simple_evaluated(*args: object) -> Union[int, str]:
            if len(args) >= 2:
                return 1
            else:
                return "x"

        def capybara(args, kwargs):
            assert_is_value(simple_evaluated(1), TypedValue(str))
            assert_is_value(simple_evaluated(1, "1"), TypedValue(int))
            assert_is_value(simple_evaluated(*args), TypedValue(str))
            assert_is_value(simple_evaluated(**kwargs), TypedValue(str))
            assert_is_value(simple_evaluated(1, y="1"), TypedValue(int))
            assert_is_value(simple_evaluated(1, **{"y": "1"}), TypedValue(int))

    @assert_passes()
    def test_is_of_type(self):
        from pyanalyze.extensions import evaluated
        from typing import Union
        from typing_extensions import Literal

        @evaluated
        def is_of_type_evaluated(x: int):
            if is_of_type(x, Literal[1]):
                return str
            else:
                return int

        def is_of_type_evaluated(x: int) -> Union[int, str]:
            if x == 1:
                return ""
            else:
                return 0

        def capybara(unannotated):
            assert_is_value(is_of_type_evaluated(1), TypedValue(str))
            assert_is_value(is_of_type_evaluated(2), TypedValue(int))
            assert_is_value(is_of_type_evaluated(unannotated), TypedValue(int))
            assert_is_value(
                is_of_type_evaluated(2 if unannotated else 1),
                TypedValue(int) | TypedValue(str),
            )

    @assert_passes()
    def test_not(self):
        from pyanalyze.extensions import evaluated
        from typing import Union
        from typing_extensions import Literal

        @evaluated
        def not_evaluated(x: int):
            if not is_of_type(x, Literal[1]):
                return str
            else:
                return int

        def not_evaluated(x: int) -> Union[int, str]:
            if x != 1:
                return ""
            else:
                return 0

        def capybara(unannotated):
            assert_is_value(not_evaluated(1), TypedValue(int))
            assert_is_value(not_evaluated(2), TypedValue(str))
            assert_is_value(not_evaluated(unannotated), TypedValue(str))
            assert_is_value(
                not_evaluated(2 if unannotated else 1),
                TypedValue(int) | TypedValue(str),
            )

    @assert_passes()
    def test_compare(self):
        from pyanalyze.extensions import evaluated
        from typing import Union

        @evaluated
        def compare_evaluated(x: object):
            if x is None:
                return str
            elif x == 1:
                return float
            else:
                return int

        def compare_evaluated(x: object) -> Union[int, str, float]:
            raise NotImplementedError

        def capybara(unannotated):
            assert_is_value(compare_evaluated(None), TypedValue(str))
            assert_is_value(compare_evaluated(1), TypedValue(float))
            assert_is_value(compare_evaluated("x"), TypedValue(int))
            assert_is_value(
                compare_evaluated(None if unannotated else 1),
                TypedValue(str) | TypedValue(float),
            )

    @assert_passes()
    def test_error(self):
        from pyanalyze.extensions import evaluated
        from typing import Any

        @evaluated
        def nonempty_please(x: str) -> int:
            if x == "":
                show_error("Non-empty string expected", argument=x)
                return Any
            else:
                return int

        def nonempty_please(x: str) -> int:
            assert x
            return len(x)

        def capybara():
            nonempty_please("")  # E: incompatible_call
            assert_is_value(nonempty_please("x"), TypedValue(int))

    @assert_passes()
    def test_restrict_kind(self):
        from pyanalyze.extensions import evaluated

        @evaluated
        def restrict_kind(x: str, y: int):
            if is_keyword(x):
                show_error("x must be positional", argument=x)
            if is_positional(y):
                show_error("y must be keyword", argument=y)
            return int

        def restrict_kind(*args, **kwargs):
            return 0

        def capybara(stuff):
            restrict_kind("x", y=1)
            restrict_kind(x="x", y=1)  # E: incompatible_call
            restrict_kind("x", 1)  # E: incompatible_call
            restrict_kind(*stuff, **stuff)
            restrict_kind(**stuff)  # E: incompatible_call
            restrict_kind(*stuff)  # E: incompatible_call

    @assert_passes()
    def test_pass(self):
        from pyanalyze.extensions import evaluated

        @evaluated
        def only_one(a: int):
            if a == 1:
                pass
            else:
                show_error("a must be 1", argument=a)
            return str

        def only_one(a: int) -> str:
            raise NotImplementedError

        def capybara():
            assert_is_value(only_one(1), TypedValue(str))
            assert_is_value(only_one(2), TypedValue(str))  # E: incompatible_call

    @assert_passes()
    def test_enum(self):
        import enum
        from pyanalyze.extensions import evaluated

        class Color(enum.Enum):
            magenta = 1
            cyan = 2

        @evaluated
        def want_enum(color: Color):
            if color is Color.magenta:
                return str
            elif color is Color.cyan:
                return int
            else:
                return bool

        def want_enum(color: Color):
            raise NotImplementedError

        def capybara(c: Color):
            assert_is_value(want_enum(Color.magenta), TypedValue(str))
            assert_is_value(want_enum(Color.cyan), TypedValue(int))
            assert_is_value(want_enum(c), TypedValue(bool))


class TestBoolOp(TestNameCheckVisitorBase):
    @assert_passes()
    def test_and(self):
        from pyanalyze.extensions import evaluated
        from typing_extensions import Literal

        @evaluated
        def use_and(a: int, b: str):
            if a == 1 and b == "x":
                return str
            return int

        def use_and(a: int, b: str) -> object:
            raise NotImplementedError

        def capybara(
            a: int, b: str, maybe_a: Literal[1, 2], maybe_b: Literal["x", "y"]
        ) -> None:
            assert_is_value(use_and(1, "x"), TypedValue(str))
            assert_is_value(use_and(a, b), TypedValue(int))
            assert_is_value(
                use_and(maybe_a, maybe_b), TypedValue(str) | TypedValue(int)
            )

    @assert_passes()
    def test_or(self):
        from pyanalyze.extensions import evaluated
        from typing_extensions import Literal

        @evaluated
        def use_or(b: str):
            if b == "x" or b == "y":
                return str
            return int

        def use_or(b: str) -> object:
            raise NotImplementedError

        def capybara(
            b: str, x_or_y: Literal["x", "y"], x_or_z: Literal["x", "z"]
        ) -> None:
            assert_is_value(use_or("x"), TypedValue(str))
            assert_is_value(use_or("y"), TypedValue(str))
            assert_is_value(use_or(b), TypedValue(int))
            assert_is_value(use_or(x_or_y), TypedValue(str))
            assert_is_value(use_or(x_or_z), TypedValue(str) | TypedValue(int))


class TestValidation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_bad(self):
        from pyanalyze.extensions import evaluated

        @evaluated
        def bad_evaluator(a: int):
            if is_of_type(a, Literal[1]):  # E: undefined_name
                print("hello")  # E: bad_evaluator
            if is_of_type():  # E: bad_evaluator
                return  # E: bad_evaluator
            if is_of_type(b, int):  # E: bad_evaluator
                return None
            if is_of_type(a, int, exclude_any=None):  # E: bad_evaluator
                return None
            if is_of_type(a, int, exclude_any=bool(a)):  # E: bad_evaluator
                return None
            if is_of_type(a, int, bad_kwarg=True):  # E: bad_evaluator
                return None
            if not_a_function():  # E: bad_evaluator
                return None
            if ~is_provided(a):  # E: bad_evaluator
                return None
            if a == 1 == a:  # E: bad_evaluator
                return None
            if a > 1:  # E: bad_evaluator
                return None
            if a == len("x"):  # E: bad_evaluator
                return None

            if is_provided("x"):  # E: bad_evaluator
                return None

            if is_provided(b):  # E: bad_evaluator
                show_error()  # E: bad_evaluator
                show_error(1)  # E: bad_evaluator
                show_error("message", argument=b)  # E: bad_evaluator
                show_error("message", arg=a)  # E: bad_evaluator
                show_error("message", argument=a)

            if (is_provided,)[0](a):  # E: bad_evaluator
                return None
            return None

        def bad_evaluator(a: int) -> None:
            pass
