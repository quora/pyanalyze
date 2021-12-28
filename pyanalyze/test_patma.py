# static analysis: ignore
from .test_node_visitor import skip_before
from .test_name_check_visitor import TestNameCheckVisitorBase


class TestPatma(TestNameCheckVisitorBase):
    @skip_before((3, 10))
    def test_singletons(self):
        self.assert_passes(
            """
            from typing import Literal
            def capybara(x: Literal[True, False, None]):
                match x:
                    case True:
                        assert_is_value(x, KnownValue(True))
                    case _:
                        assert_is_value(x, KnownValue(False) | KnownValue(None))
            """
        )

    @skip_before((3, 10))
    def test_value(self):
        self.assert_passes(
            """
            from typing import Literal

            def capybara(x: int):
                match x:
                    case None:  # E: impossible_pattern
                        assert_is_value(x, AnyValue(AnySource.unreachable))
                    case "x":  # E: impossible_pattern
                        assert_is_value(x, AnyValue(AnySource.unreachable))
                    case 3:
                        assert_is_value(x, KnownValue(3))
                    case _ if x == 4:
                        assert_is_value(x, KnownValue(4))
                    case _:
                        assert_is_value(x, TypedValue(int))
            """
        )

    @skip_before((3, 10))
    def test_sequence(self):
        self.assert_passes(
            """
            import collections.abc
            from typing import Tuple

            def capybara(seq: Tuple[int, ...], obj: object):
                match seq:
                    case [1, 2, 3]:
                        assert_is_value(
                            seq,
                            SequenceIncompleteValue(
                                tuple,
                                [TypedValue(int), TypedValue(int), TypedValue(int)]
                            )
                        )
                    case [1, *x]:
                        assert_is_value(x, GenericValue(list, [TypedValue(int)]))

                match obj:
                    case [*x]:
                        assert_is_value(
                            obj,
                            TypedValue(collections.abc.Sequence),
                            skip_annotated=True
                        )
                        assert_is_value(x, AnyValue(AnySource.generic_argument))

                assert_is_value(seq[0], TypedValue(int))
                match seq[0]:
                    case [1, 2, 3]:  # E: impossible_pattern
                        pass
            """
        )
