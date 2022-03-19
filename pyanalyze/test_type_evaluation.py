# static analysis: ignore
from .value import AnySource, AnyValue, KnownValue, TypedValue, assert_is_value
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

    @assert_passes()
    def test_platform(self):
        from pyanalyze.extensions import evaluated
        import sys
        from typing_extensions import Literal

        @evaluated
        def where_am_i():
            if sys.platform == "darwin":
                return Literal["On a Mac"]
            else:
                return Literal["Somewhere else"]

        def where_am_i():
            raise NotImplementedError

        expected = "On a Mac" if sys.platform == "darwin" else "Somewhere else"

        def capybara():
            assert_is_value(where_am_i(), KnownValue(expected))

    @assert_passes()
    def test_version(self):
        from pyanalyze.extensions import evaluated
        import sys
        from typing_extensions import Literal

        @evaluated
        def is_walrus_available():
            if sys.version_info >= (3, 8):
                return Literal[True]
            return Literal[False]

        def is_walrus_available():
            return sys.version_info >= (3, 8)

        expected = is_walrus_available()

        def capybara():
            assert_is_value(is_walrus_available(), KnownValue(expected))

    @assert_passes()
    def test_nested_ifs(self):
        from pyanalyze.extensions import evaluated, is_of_type
        from typing_extensions import Literal

        @evaluated
        def is_int(i: int):
            if is_of_type(i, Literal[1, 2]):
                if i == 1:
                    return Literal[1]
                elif i == 2:
                    return Literal[2]
            return Literal[3]

        def capybara():
            assert_is_value(is_int(1), KnownValue(1))

    @assert_passes()
    def test_not_equals(self):
        from pyanalyze.extensions import evaluated

        @evaluated
        def want_one(x: int = 1, y: bool = True):
            if x != 1:
                show_error("want one", argument=x)
            if y is not True:
                show_error("want one", argument=y)
            return None

        def want_one(x: int = 1, y: bool = True) -> None:
            pass

        def capybara():
            want_one(2)  # E: incompatible_call
            want_one(y=False)  # E: incompatible_call

    @assert_passes()
    def test_reveal_type(self):
        from pyanalyze.extensions import evaluated

        @evaluated
        def has_default(x: int = 1):
            reveal_type(x)
            return None

        def has_default(x: int = 1) -> None:
            pass

        def capybara(i: int):
            has_default()  # E: incompatible_call
            has_default(i)  # E: incompatible_call

    @assert_passes()
    def test_return(self):
        from pyanalyze.extensions import evaluated

        @evaluated
        def maybe_use_header(x: bool) -> int:
            if x is True:
                return str

        def capybara(x: bool):
            assert_is_value(maybe_use_header(True), TypedValue(str))
            assert_is_value(maybe_use_header(x), TypedValue(int))

    @assert_passes()
    def test_generic(self):
        from pyanalyze.extensions import evaluated
        from typing import TypeVar

        T1 = TypeVar("T1")

        @evaluated
        def identity(x: T1):
            return T1

        @evaluated
        def identity2(x: T1) -> T1:
            pass

        def capybara(unannotated):
            assert_is_value(identity(1), KnownValue(1))
            assert_is_value(identity(unannotated), AnyValue(AnySource.unannotated))
            assert_is_value(identity2(1), KnownValue(1))
            assert_is_value(identity2(unannotated), AnyValue(AnySource.unannotated))


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

    @assert_passes()
    def test_literal_or(self):
        from pyanalyze.extensions import evaluated
        from typing import Union

        @evaluated
        def is_one(i: int):
            if i == 1 or i == -1:
                show_error("bad argument", argument=i)
                return int
            return str

        def is_one(i: int) -> Union[int, str]:
            raise NotImplementedError

        def capybara():
            val = is_one(-1)  # E: incompatible_call
            assert_is_value(val, TypedValue(int))
            assert_is_value(is_one(2), TypedValue(str))


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


class TestExamples(TestNameCheckVisitorBase):
    @assert_passes()
    def test_open(self):
        from pyanalyze.extensions import evaluated, is_of_type
        from typing import Callable, Union, IO, BinaryIO, Any, Optional
        from typing_extensions import Literal
        from io import (
            BufferedRandom,
            BufferedReader,
            BufferedWriter,
            FileIO,
            TextIOWrapper,
        )

        _OpenFile = Union[str, bytes, int]
        _Opener = Callable[[str, int], int]

        # These are simplified
        OpenTextModeUpdating = Literal["r+", "w+", "a+", "x+"]
        OpenTextModeWriting = Literal["w", "wt", "tw", "a", "at", "ta", "x", "xt", "tx"]
        OpenTextModeReading = Literal["r", "rt", "tr"]
        OpenTextMode = Union[
            OpenTextModeUpdating, OpenTextModeWriting, OpenTextModeReading
        ]
        OpenBinaryModeUpdating = Literal["rb+", "wb+", "ab+", "xb+"]
        OpenBinaryModeWriting = Literal["wb", "bw", "ab", "ba", "xb", "bx"]
        OpenBinaryModeReading = Literal["rb", "br"]
        OpenBinaryMode = Union[
            OpenBinaryModeUpdating, OpenBinaryModeReading, OpenBinaryModeWriting
        ]

        @evaluated
        def open2(
            file: _OpenFile,
            mode: str = "r",
            buffering: int = -1,
            encoding: Optional[str] = None,
            errors: Optional[str] = None,
            newline: Optional[str] = None,
            closefd: bool = False,
            opener: Optional[_Opener] = None,
        ) -> IO[Any]:
            if is_of_type(mode, OpenTextMode):
                return TextIOWrapper
            elif is_of_type(mode, OpenBinaryMode):
                if encoding is not None:
                    show_error(
                        "'encoding' argument may not be provided in binary moode",
                        argument=encoding,
                    )
                if errors is not None:
                    show_error(
                        "'errors' argument may not be provided in binary moode",
                        argument=errors,
                    )
                if newline is not None:
                    show_error(
                        "'newline' argument may not be provided in binary moode",
                        argument=newline,
                    )
                if buffering == 0:
                    return FileIO
                elif buffering == -1 or buffering == 1:
                    if is_of_type(mode, OpenBinaryModeUpdating):
                        return BufferedRandom
                    elif is_of_type(mode, OpenBinaryModeWriting):
                        return BufferedWriter
                    elif is_of_type(mode, OpenBinaryModeReading):
                        return BufferedReader

                # Buffering cannot be determined: fall back to BinaryIO
                return BinaryIO
            # Fallback if mode is not specified
            return IO[Any]

        def capybara():
            assert_is_value(open2("x", "r"), TypedValue(TextIOWrapper))
            open2("x", "rb", encoding="utf-8")  # E: incompatible_call
            assert_is_value(open2("x", "rb", buffering=0), TypedValue(FileIO))
            assert_is_value(open2("x", "rb+"), TypedValue(BufferedRandom))
            assert_is_value(open2("x", "rb"), TypedValue(BufferedReader))
            assert_is_value(open2("x", "rb", buffering=1), TypedValue(BufferedReader))

    @assert_passes()
    def test_safe_upcast(self):
        from typing import Type, Any, TypeVar
        from pyanalyze.extensions import evaluated, show_error, is_of_type

        T1 = TypeVar("T1")

        @evaluated
        def safe_upcast(typ: Type[T1], value: object):
            if is_of_type(value, T1):
                return T1
            show_error("unsafe cast")
            return Any

        def capybara():
            assert_is_value(safe_upcast(object, 1), TypedValue(object))
            assert_is_value(safe_upcast(int, 1), TypedValue(int))
            safe_upcast(str, 1)  # E: incompatible_call

    @assert_passes()
    def test_safe_contains(self):
        from typing import List, TypeVar, Container
        from pyanalyze.extensions import evaluated, show_error, is_of_type

        T1 = TypeVar("T1")
        T2 = TypeVar("T2")

        @evaluated
        def safe_contains(elt: T1, container: Container[T2]) -> bool:
            if not is_of_type(elt, T2) and not is_of_type(container, Container[T1]):
                show_error("Element cannot be a member of container")

        def capybara(lst: List[int], o: object):
            safe_contains(True, ["x"])  # E: incompatible_call
            safe_contains("x", lst)  # E: incompatible_call
            safe_contains(True, lst)
            safe_contains(o, lst)
