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
                        assert_is_value(
                            x,
                            GenericValue(list, [AnyValue(AnySource.generic_argument)])
                        )

                assert_is_value(seq[0], TypedValue(int))
                match seq[0]:
                    case [1, 2, 3]:  # E: impossible_pattern
                        pass
            """
        )

    @skip_before((3, 10))
    def test_or(self):
        self.assert_passes(
            """
            import collections.abc
            from typing import Tuple

            def capybara(obj: object):
                match obj:
                    case 1 | 2:
                        assert_is_value(obj, KnownValue(1) | KnownValue(2))
                    case (3 as x) | (4 as x):
                        assert_is_value(x, KnownValue(3) | KnownValue(4))
            """
        )

    @skip_before((3, 10))
    def test_mapping(self):
        self.assert_passes(
            """
            import collections.abc
            from typing import Tuple

            def capybara(obj: object):
                match {1: 2, 3: 4, 5: 6}:
                    case {1: x}:
                        assert_is_value(x, KnownValue(2))
                    case {3: 4, **x}:
                        assert_is_value(x, DictIncompleteValue(
                            dict, [
                                KVPair(KnownValue(1), KnownValue(2)),
                                KVPair(KnownValue(5), KnownValue(6)),
                            ]
                        ))
            """
        )

    @skip_before((3, 10))
    def test_class_pattern(self):
        self.assert_passes(
            """
            import collections.abc
            from typing import Tuple

            class NotMatchable:
                x: str

            class MatchArgs:
                __match_args__ = ("x", "y")
                x: str
                y: int

            def capybara(obj: object):
                match obj:
                    case int(1, 2):  # E: bad_match
                        pass
                    case int(2):
                        assert_is_value(obj, KnownValue(2))
                    case int("x"):  # E: impossible_pattern
                        pass
                    case int():
                        assert_is_value(obj, TypedValue(int))
                    case NotMatchable(x="x"):
                        pass
                    case NotMatchable("x"):  # E: bad_match
                        pass
                    case NotMatchable():
                        pass
                    case MatchArgs("x", 1 as y):
                        assert_is_value(y, KnownValue(1))
                    case MatchArgs(x) if x == "x":
                        assert_is_value(x, KnownValue("x"))
                    case MatchArgs(x):
                        assert_is_value(x, TypedValue(str))
                    case MatchArgs("x", x="x"): # E: bad_match
                        pass
                    case MatchArgs(1, 2, 3):  # E: bad_match
                        pass
            """
        )

    @skip_before((3, 10))
    def test_bool_narrowing(self):
        self.assert_passes(
            """
            class X:
                true = True

            def capybara(b: bool):
                match b:
                    # Make sure we hit the MatchValue case, not MatchSingleton
                    case X.true:
                        assert_is_value(b, KnownValue(True))
                    case _ as b2:
                        assert_is_value(b, KnownValue(False))
                        assert_is_value(b2, KnownValue(False))
            """
        )
        self.assert_passes(
            """
            def capybara(b: bool):
                match b:
                    case True:
                        assert_is_value(b, KnownValue(True))
                    case _ as b2:
                        assert_is_value(b, KnownValue(False))
                        assert_is_value(b2, KnownValue(False))
            """
        )

    @skip_before((3, 10))
    def test_enum_narrowing(self):
        self.assert_passes(
            """
            from enum import Enum

            class Planet(Enum):
                mercury = 1
                venus = 2
                earth = 3

            def capybara(p: Planet):
                match p:
                    case Planet.mercury:
                        assert_is_value(p, KnownValue(Planet.mercury))
                    case Planet.venus:
                        assert_is_value(p, KnownValue(Planet.venus))
                    case _ as p2:
                        assert_is_value(p2, KnownValue(Planet.earth))
                        assert_is_value(p, KnownValue(Planet.earth))
            """
        )
