# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import TypedDictValue, TypedValue, AnyValue, AnySource


class TestExtraKeys(TestNameCheckVisitorBase):
    @assert_passes()
    def test_signature(self):
        from pyanalyze.extensions import has_extra_keys
        from typing_extensions import TypedDict

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        def capybara() -> None:
            x = TD(a="a", b=1)
            assert_is_value(
                x,
                TypedDictValue(
                    {"a": (True, TypedValue(str))}, extra_keys=TypedValue(int)
                ),
            )

            TD(a="a", b="b")  # E: incompatible_argument

    @assert_passes()
    def test_methods(self):
        from pyanalyze.extensions import has_extra_keys
        from typing_extensions import TypedDict, assert_type, Literal
        from typing import Union

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
        from pyanalyze.extensions import has_extra_keys
        from typing_extensions import TypedDict, Unpack, assert_type

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
        from pyanalyze.extensions import has_extra_keys
        from typing_extensions import TypedDict
        from typing import Any, Dict

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        @has_extra_keys(bool)
        class TD2(TypedDict):
            a: str

        @has_extra_keys(bytes)
        class TD3(TypedDict):
            a: str

        def want_td(td: TD) -> None:
            pass

        def capybara(td: TD, td2: TD2, td3: TD3, anydict: Dict[str, Any]) -> None:
            want_td(td)
            want_td(td2)
            want_td(td3)  # E: incompatible_argument
            want_td(anydict)

    @assert_passes()
    def test_iteration(self):
        from pyanalyze.extensions import has_extra_keys
        from typing_extensions import TypedDict, assert_type, Literal
        from typing import Union

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
                assert_type(k, Literal["a"])


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
                    {"a": (True, TypedValue(int)), "b": (True, TypedValue(str))}
                ),
            )
            assert_is_value(x["a"], TypedValue(int))
            assert_is_value(
                y,
                TypedDictValue(
                    {"a": (True, TypedValue(int)), "b": (True, TypedValue(str))}
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
