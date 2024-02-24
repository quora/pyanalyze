# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import AnySource, AnyValue, TypedDictEntry, TypedDictValue, TypedValue


class TestExtraKeys(TestNameCheckVisitorBase):
    @assert_passes()
    def test_signature(self):
        from typing_extensions import TypedDict

        from pyanalyze.extensions import has_extra_keys

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        def capybara() -> None:
            x = TD(a="a", b=1)
            assert_is_value(
                x,
                TypedDictValue(
                    {"a": TypedDictEntry(TypedValue(str))}, extra_keys=TypedValue(int)
                ),
            )

            TD(a="a", b="b")  # E: incompatible_argument

    @assert_passes()
    def test_methods(self):
        from typing import Union

        from typing_extensions import Literal, TypedDict, assert_type

        from pyanalyze.extensions import has_extra_keys

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        class NormalTD(TypedDict):
            a: str

        def getitem(td: TD, ntd: NormalTD) -> None:
            td["b"] = 3
            ntd["b"] = 3  # E: invalid_typeddict_key

        def setitem(td: TD) -> None:
            assert_type(td["b"], int)

        def get(td: TD) -> None:
            assert_type(td.get("b", "x"), Union[int, Literal["x"]])

        def pop(td: TD) -> None:
            assert_type(td.pop("b"), int)

        def setdefault(td: TD) -> None:
            assert_type(td.setdefault("b", "x"), Union[int, Literal["x"]])

    @assert_passes()
    def test_kwargs_annotation(self):
        from typing_extensions import TypedDict, Unpack, assert_type

        from pyanalyze.extensions import has_extra_keys

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        def caller(**kwargs: Unpack[TD]) -> None:
            assert_type(kwargs["b"], int)

        def capybara() -> None:
            caller(a="x", b=1)
            caller(a="x", b="y")  # E: incompatible_argument

    @assert_passes()
    def test_compatibility(self):
        from typing import Any, Dict

        from typing_extensions import ReadOnly, TypedDict

        from pyanalyze.extensions import has_extra_keys

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        @has_extra_keys(bool)
        class TD2(TypedDict):
            a: str

        @has_extra_keys(bytes)
        class TD3(TypedDict):
            a: str

        @has_extra_keys(ReadOnly[int])
        class TD4(TypedDict):
            a: str

        def want_td(td: TD) -> None:
            pass

        def want_td4(td: TD4) -> None:
            pass

        def capybara(td: TD, td2: TD2, td3: TD3, anydict: Dict[str, Any]) -> None:
            want_td(td)
            want_td(td2)  # E: incompatible_argument
            want_td(td3)  # E: incompatible_argument
            want_td(anydict)

            want_td4(td)
            want_td4(td2)
            want_td4(td3)  # E: incompatible_argument
            want_td4(anydict)

    @assert_passes()
    def test_iteration(self):
        from typing import Union

        from typing_extensions import Literal, TypedDict, assert_type

        from pyanalyze.extensions import has_extra_keys

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        class TD2(TypedDict):
            a: str

        def capybara(td: TD, td2: TD2) -> None:
            for k, v in td.items():
                assert_type(k, str)
                assert_type(v, Union[int, str])
            for k in td:
                assert_type(k, Union[str, Literal["a"]])

            for k, v in td2.items():
                assert_type(k, str)
                assert_type(v, str)
            for k in td2:
                assert_type(k, Union[str, Literal["a"]])


class TestTypedDict(TestNameCheckVisitorBase):
    @assert_passes()
    def test_constructor(self):
        from typing_extensions import NotRequired, TypedDict

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
                    {
                        "x": TypedDictEntry(TypedValue(int)),
                        "y": TypedDictEntry(TypedValue(str)),
                    }
                ),
            )
            Capybara(x=1)  # E: incompatible_call

            maybe_cap = MaybeCapybara(x=1)
            assert_is_value(
                maybe_cap,
                TypedDictValue(
                    {
                        "x": TypedDictEntry(TypedValue(int)),
                        "y": TypedDictEntry(TypedValue(str), required=False),
                    }
                ),
            )

    @assert_passes()
    def test_unknown_key(self):
        from typing_extensions import TypedDict, assert_type

        class Capybara(TypedDict):
            x: int

        def user(c: Capybara):
            assert_type(c["x"], int)
            c["y"]  # E: invalid_typeddict_key

    @assert_passes()
    def test_basic(self):
        from mypy_extensions import TypedDict as METypedDict
        from typing_extensions import TypedDict as TETypedDict

        T = METypedDict("T", {"a": int, "b": str})
        T2 = TETypedDict("T2", {"a": int, "b": str})

        def capybara(x: T, y: T2):
            assert_is_value(
                x,
                TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int)),
                        "b": TypedDictEntry(TypedValue(str)),
                    }
                ),
            )
            assert_is_value(x["a"], TypedValue(int))
            assert_is_value(
                y,
                TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int)),
                        "b": TypedDictEntry(TypedValue(str)),
                    }
                ),
            )
            assert_is_value(y["a"], TypedValue(int))

    @assert_passes()
    def test_unknown_key_unresolved(self):
        from mypy_extensions import TypedDict

        T = TypedDict("T", {"a": int, "b": str})

        def capybara(x: T):
            val = x["not a key"]  # E: invalid_typeddict_key
            assert_is_value(val, AnyValue(AnySource.error))

    @assert_passes()
    def test_invalid_key(self):
        from mypy_extensions import TypedDict

        T = TypedDict("T", {"a": int, "b": str})

        def capybara(x: T):
            x[0]  # E: invalid_typeddict_key

    @assert_passes()
    def test_total(self):
        from typing_extensions import TypedDict

        class TD(TypedDict, total=False):
            a: int
            b: str

        class TD2(TD):
            c: float

        def f(td: TD) -> None:
            pass

        def g(td2: TD2) -> None:
            pass

        def caller() -> None:
            f({})
            f({"a": 1})
            f({"a": 1, "b": "c"})
            f({"a": "a"})  # E: incompatible_argument
            g({"c": 1.0})
            g({})  # E: incompatible_argument


class TestReadOnly(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        from typing import Any, Dict

        from typing_extensions import NotRequired, ReadOnly, TypedDict

        class TD(TypedDict):
            a: ReadOnly[NotRequired[int]]
            b: ReadOnly[str]

        def capybara(td: TD, anydict: Dict[str, Any]) -> None:
            td["a"] = 1  # E: readonly_typeddict
            td["b"] = "a"  # E: readonly_typeddict
            td.update(anydict)  # E: invalid_typeddict_key
            td.setdefault("a", 1)  # E: readonly_typeddict
            td.setdefault("b", "a")  # E: readonly_typeddict
            td.pop("a")  # E: readonly_typeddict
            td.pop("b")  # E: incompatible_argument
            del td["a"]  # E: readonly_typeddict
            del td["b"]  # E: readonly_typeddict

    @assert_passes()
    def test_compatibility(self):
        from typing import Any, Dict

        from typing_extensions import ReadOnly, TypedDict

        class TD(TypedDict):
            a: int

        class TD2(TypedDict):
            a: ReadOnly[int]

        class TD3(TypedDict):
            a: bool

        class TD4(TypedDict):
            a: str

        def want_td(td: TD) -> None:
            pass

        def want_td2(td: TD2) -> None:
            pass

        def capybara(
            td: TD, td2: TD2, td3: TD3, td4: TD4, anydict: Dict[str, Any]
        ) -> None:
            want_td(td)
            want_td(td2)  # E: incompatible_argument
            want_td(td3)  # E: incompatible_argument
            want_td(td4)  # E: incompatible_argument
            want_td(anydict)

            want_td2(td)
            want_td2(td2)
            want_td2(td3)
            want_td2(td4)  # E: incompatible_argument
            want_td2(anydict)


class TestClosed(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        from typing import Any, Dict

        from typing_extensions import NotRequired, TypedDict

        class Closed(TypedDict, closed=True):
            a: NotRequired[int]
            b: str

        class Open(TypedDict):
            a: NotRequired[int]
            b: str

        def want_closed(td: Closed) -> None:
            pass

        def want_open(td: Open) -> None:
            pass

        def capybara(closed: Closed, open: Open, anydict: Dict[str, Any]) -> None:
            closed["a"] = 1
            closed["b"] = "a"
            closed["a"] = "x"  # E: incompatible_argument

            open["a"] = 1
            open["b"] = "a"
            open["a"] = "x"  # E: incompatible_argument

            closed.update(anydict)  # E: invalid_typeddict_key
            open.update(anydict)  # E: invalid_typeddict_key

            x: Closed = {"a": 1, "b": "a", "c": "x"}  # E: incompatible_assignment
            y: Open = {"a": 1, "b": "a", "c": "x"}

            want_closed(closed)
            want_closed(open)  # E: incompatible_argument

            want_open(open)
            want_open(closed)
