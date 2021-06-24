from qcore.asserts import assert_eq, assert_is

from .operations import concrete_values_from_iterable
from .test_name_check_visitor import VISITOR as CTX
from .value import (
    GenericValue,
    KnownValue,
    TypedValue,
    MultiValuedValue,
    SequenceIncompleteValue,
)


def test_concrete_values_from_iterable() -> None:
    assert_is(None, concrete_values_from_iterable(KnownValue(1), CTX))
    assert_eq((), concrete_values_from_iterable(KnownValue(()), CTX))
    assert_eq(
        (KnownValue(1), KnownValue(2)),
        concrete_values_from_iterable(KnownValue((1, 2)), CTX),
    )
    assert_eq(
        MultiValuedValue((KnownValue(1), KnownValue(2))),
        concrete_values_from_iterable(
            SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]), CTX
        ),
    )
    assert_eq(
        TypedValue(int),
        concrete_values_from_iterable(GenericValue(list, [TypedValue(int)]), CTX),
    )
    assert_eq(
        MultiValuedValue([KnownValue(1), KnownValue(3), KnownValue(2), KnownValue(4)]),
        concrete_values_from_iterable(
            MultiValuedValue(
                [
                    SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]),
                    KnownValue((3, 4)),
                ]
            ),
            CTX,
        ),
    )
    assert_eq(
        MultiValuedValue([KnownValue(1), KnownValue(2), TypedValue(int)]),
        concrete_values_from_iterable(
            MultiValuedValue(
                [
                    SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]),
                    GenericValue(list, [TypedValue(int)]),
                ]
            ),
            CTX,
        ),
    )
    assert_eq(
        MultiValuedValue([KnownValue(1), KnownValue(2), KnownValue(3)]),
        concrete_values_from_iterable(
            MultiValuedValue(
                [
                    SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]),
                    KnownValue((3,)),
                ]
            ),
            CTX,
        ),
    )
