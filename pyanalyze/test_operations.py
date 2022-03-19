# static analysis: ignore
from .implementation import assert_is_value
from .value import AnySource, AnyValue, KnownValue, MultiValuedValue, TypedValue
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .error_code import ErrorCode


class TestBinOps(TestNameCheckVisitorBase):
    @assert_passes()
    def test_binop(self):
        from typing import Union

        def tucotuco():
            assert_is_value(2 + 3, KnownValue(5))

        def capybara(x: Union[int, float], y: Union[int, float]) -> float:
            return x + y

    @assert_passes()
    def test_inplace_binop(self):
        class Capybara:
            def __add__(self, x: int) -> str:
                return ""

            def __iadd__(self, x: str) -> int:
                return 0

        def tucotuco():
            x = Capybara()
            assert_is_value(x + 1, TypedValue(str))
            x += "a"
            assert_is_value(x, TypedValue(int))

    @assert_passes()
    def test_binop_notimplemented(self):
        from pyanalyze.extensions import assert_type

        class Capybara:
            def __add__(self, x: str) -> bool:
                if not isinstance(x, str):
                    return NotImplemented
                return len(x) > 3

        def pacarana():
            assert_type(Capybara() + "x", bool)


class TestBoolOp(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def capybara(x):
            if x:
                cond = str(x)
                cond2 = True
            else:
                cond = None
                cond2 = None
            assert_is_value(cond, MultiValuedValue([TypedValue(str), KnownValue(None)]))
            assert_is_value(
                cond2, MultiValuedValue([KnownValue(True), KnownValue(None)])
            )
            assert_is_value(
                cond and 1,
                MultiValuedValue([TypedValue(str), KnownValue(None), KnownValue(1)]),
                skip_annotated=True,
            )
            assert_is_value(
                cond2 and 1,
                MultiValuedValue([KnownValue(None), KnownValue(1)]),
                skip_annotated=True,
            )
            assert_is_value(
                cond or 1,
                MultiValuedValue([TypedValue(str), KnownValue(1)]),
                skip_annotated=True,
            )
            assert_is_value(
                cond2 or 1,
                MultiValuedValue([KnownValue(True), KnownValue(1)]),
                skip_annotated=True,
            )

        def hutia(x=None):
            assert_is_value(x, AnyValue(AnySource.unannotated) | KnownValue(None))
            assert_is_value(
                x or 1,
                AnyValue(AnySource.unannotated) | KnownValue(1),
                skip_annotated=True,
            )
            y = x or 1
            assert_is_value(
                y, AnyValue(AnySource.unannotated) | KnownValue(1), skip_annotated=True
            )
            assert_is_value(
                (True if x else False) or None, KnownValue(True) | KnownValue(None)
            )


class TestOperators(TestNameCheckVisitorBase):
    @assert_passes(settings={ErrorCode.value_always_true: False})
    def test_not(self):
        def capybara(x):
            assert_is_value(not x, TypedValue(bool), skip_annotated=True)
            assert_is_value(not True, KnownValue(False))

    @assert_passes()
    def test_unary_op(self):
        def capybara(x):
            assert_is_value(~x, AnyValue(AnySource.from_another))
            assert_is_value(~3, KnownValue(-4))

    @assert_passes()
    def test_binop_type_inference(self):
        def capybara(x):
            assert_is_value(1 + int(x), TypedValue(int))
            assert_is_value(3 * int(x), TypedValue(int))
            assert_is_value("foo" + str(x), TypedValue(str))
            assert_is_value(1 + float(x), TypedValue(float))
            assert_is_value(1.0 + int(x), TypedValue(float))
            assert_is_value(3 * 3.0 + 1, KnownValue(10.0))

    @assert_passes()
    def test_union(self):
        from typing import Union

        def capybara(x: Union[int, str]) -> None:
            assert_is_value(x * 3, MultiValuedValue([TypedValue(int), TypedValue(str)]))

    @assert_passes()
    def test_rop(self):
        class HasAdd:
            def __add__(self, other: int) -> "HasAdd":
                raise NotImplementedError

        class HasRadd:
            def __radd__(self, other: int) -> "HasRadd":
                raise NotImplementedError

        class HasBoth:
            def __add__(self, other: "HasBoth") -> "HasBoth":
                raise NotImplementedError

            def __radd__(self, other: "HasBoth") -> int:
                raise NotImplementedError

        def capybara(x):
            ha = HasAdd()
            hr = HasRadd()
            assert_is_value(1 + hr, TypedValue(HasRadd))
            assert_is_value(x + hr, AnyValue(AnySource.from_another))
            assert_is_value(ha + 1, TypedValue(HasAdd))
            assert_is_value(ha + x, AnyValue(AnySource.from_another))
            assert_is_value(HasBoth() + HasBoth(), TypedValue(HasBoth))

    @assert_passes()
    def test_unsupported_unary_op(self):
        def capybara():
            ~"capybara"  # E: unsupported_operation

    @assert_passes()
    def test_int_float_product(self):
        def capybara(f: float, i: int):
            assert_is_value(i * f, TypedValue(float))

    @assert_passes()
    def test_contains(self):
        from pyanalyze.extensions import assert_type

        def capybara(x, y):
            assert_type(x in y, bool)


class TestAugAssign(TestNameCheckVisitorBase):
    @assert_passes()
    def test_aug_assign(self):
        def capybara(condition):
            x = 1
            x += 2
            assert_is_value(x, KnownValue(3))


class TestCompare(TestNameCheckVisitorBase):
    @assert_passes()
    def test_multi(self):
        from typing_extensions import Literal

        def capybara(i: Literal[1, 2, 3], x: Literal[3, 4]) -> None:
            assert_is_value(i, KnownValue(1) | KnownValue(2) | KnownValue(3))
            assert_is_value(x, KnownValue(3) | KnownValue(4))
            if 1 < i < 3 != x:
                assert_is_value(i, KnownValue(2))
                assert_is_value(x, KnownValue(4))


class TestAdd(TestNameCheckVisitorBase):
    @assert_passes()
    def test_bytes_and_text(self):
        def capybara():
            return b"foo" + "bar"  # E: unsupported_operation

    @assert_passes()
    def test_text_and_bytes(self):
        def capybara():
            return "foo" + b"bar"  # E: unsupported_operation
