# static analysis: ignore
from __future__ import absolute_import, division, print_function, unicode_literals
from qcore.asserts import assert_eq, assert_ge
import six

from .error_code import ErrorCode
from .format_strings import (
    ConversionSpecifier,
    PercentFormatString,
    StarConversionSpecifier,
    FormatString,
    ReplacementField,
    IndexOrAttribute,
    FormatSpec,
    parse_format_string,
)
from .value import (
    KnownValue,
    UNRESOLVED_VALUE,
    DictIncompleteValue,
    SequenceIncompleteValue,
    TypedValue,
)

from .test_node_visitor import assert_passes, assert_fails, only_before, skip_before
from .test_name_check_visitor import TestNameCheckVisitorBase


PERCENT_TESTCASES = [
    ("%(a)s", (ConversionSpecifier("s", mapping_key="a"),), ("", "")),
    # all the things
    (
        "%(a)#0- +9000.9000Ld",
        (
            ConversionSpecifier(
                "d",
                mapping_key="a",
                conversion_flags="#0- +",
                field_width=9000,
                precision=9000,
                length_modifier="L",
            ),
        ),
        ("", ""),
    ),
    # * for int fields
    ("%*.*s", (ConversionSpecifier("s", field_width="*", precision="*"),), ("", "")),
    # bytes patterns
    (b"%s", (ConversionSpecifier("s", is_bytes=True),), (b"", b"")),
    (
        b"%(a)#0- +9000.9000Ld",
        (
            ConversionSpecifier(
                "d",
                mapping_key="a",
                conversion_flags="#0- +",
                field_width=9000,
                precision=9000,
                length_modifier="L",
                is_bytes=True,
            ),
        ),
        (b"", b""),
    ),
    # pieces in between
    ("a%sb", (ConversionSpecifier("s"),), ("a", "b")),
    ("a%sb%dc", (ConversionSpecifier("s"), ConversionSpecifier("d")), ("a", "b", "c")),
    # no specifiers
    ("abc", (), ("abc",)),
]


def test_parse_percent():
    for pattern, specifiers, raw_pieces in PERCENT_TESTCASES:
        assert_eq(
            PercentFormatString(
                pattern,
                isinstance(pattern, bytes),
                specifiers=specifiers,
                raw_pieces=raw_pieces,
            ),
            PercentFormatString.from_pattern(pattern),
        )


DOT_FORMAT_TESTCASES = [
    ("", []),
    ("a", ["a"]),
    ("{}", [ReplacementField(None)]),
    ("{1}", [ReplacementField(1)]),
    ("{a}", [ReplacementField("a")]),
    (
        "{a.b}",
        [ReplacementField("a", index_attribute=((IndexOrAttribute.attribute, "b"),))],
    ),
    (
        "{a[0]}",
        [ReplacementField("a", index_attribute=((IndexOrAttribute.index, "0"),))],
    ),
    (
        "{a[}]}",
        [ReplacementField("a", index_attribute=((IndexOrAttribute.index, "}"),))],
    ),
    (
        "{a.b[0].c}",
        [
            ReplacementField(
                "a",
                index_attribute=(
                    (IndexOrAttribute.attribute, "b"),
                    (IndexOrAttribute.index, "0"),
                    (IndexOrAttribute.attribute, "c"),
                ),
            )
        ],
    ),
    ("{a!r}", [ReplacementField("a", conversion="r")]),
    ("{a:3}", [ReplacementField("a", format_spec=FormatSpec(["3"]))]),
    (
        "{a:{b}}",
        [ReplacementField("a", format_spec=FormatSpec([ReplacementField("b")]))],
    ),
    (
        "{a.b!r:{c:{d}}}",
        [
            ReplacementField(
                "a",
                index_attribute=((IndexOrAttribute.attribute, "b"),),
                conversion="r",
                format_spec=FormatSpec(
                    [
                        ReplacementField(
                            "c", format_spec=FormatSpec([ReplacementField("d")])
                        )
                    ]
                ),
            )
        ],
    ),
    ("{} {}", [ReplacementField(None), " ", ReplacementField(None)]),
    ("{{ }}", ["{ }"]),
]
DOT_FORMAT_ERRORS = [
    ("{", 2, "expected '}' before end of string"),
    ("}", 1, "single '}' encountered in format string"),
    ("{x!c}", 4, "Unknown conversion specifier 'c'"),
    ("{x.3}", 4, "invalid attribute '3'"),
    ("{x.a", 4, "expected '}' before end of string"),
    ("{x[1}", 6, "expected ']' before end of string"),
    ("{x!r.a}", 4, "expected one of ':', '}'"),
    ("{x!r:a", 7, "expected '}' before end of string"),
]


def test_parse_format_string():
    for format_string, expected in DOT_FORMAT_TESTCASES:
        assert_eq(
            (FormatString(expected), []),
            parse_format_string(format_string),
            extra=format_string,
        )
    for format_string, position, message in DOT_FORMAT_ERRORS:
        _, errors = parse_format_string(format_string)
        assert_ge(len(errors), 1)
        assert_eq((position, message), errors[0], extra=format_string)


def assert_lints(pattern, errors):
    """Asserts that linting this pattern produces the given errors."""
    fs = PercentFormatString.from_pattern(pattern)
    assert_eq(errors, list(fs.lint()), extra="while linting {}".format(pattern))


def test_lint():
    assert_lints("%s", [])
    assert_lints(
        "%.1%", ["using % combined with optional specifiers does not make sense"]
    )
    assert_lints(
        "%(a)s%s",
        ["cannot combine specifiers that require a mapping with those that do not"],
    )
    assert_lints(
        "%(a)*d",
        ["cannot combine specifiers that require a mapping with those that do not"],
    )
    assert_lints(
        "%(a).*d",
        ["cannot combine specifiers that require a mapping with those that do not"],
    )
    assert_lints("%k", ["invalid conversion specifier in %k"])
    assert_lints(
        "%b", ["the %b conversion specifier works only on Python 3 bytes patterns"]
    )
    if six.PY2:
        assert_lints(
            b"%b", ["the %b conversion specifier works only on Python 3 bytes patterns"]
        )
        assert_lints("%a", ["the %a conversion specifier works only in Python 3"])
        assert_lints(b"%a", ["the %a conversion specifier works only in Python 3"])


class TestAccept(object):
    def assert_errors(self, obj, arg, expected):
        actual = list(obj.accept(arg))
        if len(actual) != len(expected):
            assert False, "did not get the expected number of errors: {} vs. {}".format(
                expected, actual
            )
        for actual_err, expected_err in zip(actual, expected):
            # allow the user to pass in a prefix in case the error ends with extra information that
            # we don't want to test for (e.g. the repr of a Value object)
            assert actual_err.startswith(expected_err), "{} does not match {}".format(
                actual_err, expected_err
            )

    def test_conversion_specifier(self):
        self.assert_errors(ConversionSpecifier("d"), UNRESOLVED_VALUE, [])
        self.assert_errors(
            ConversionSpecifier("d"),
            KnownValue("string"),
            ["%d conversion specifier accepts numbers"],
        )
        self.assert_errors(ConversionSpecifier("d"), KnownValue(3), [])
        self.assert_errors(ConversionSpecifier("r"), KnownValue(3), [])
        self.assert_errors(ConversionSpecifier("a"), KnownValue(3), [])

        # %%
        self.assert_errors(
            ConversionSpecifier("%"), KnownValue(3), ["%% does not accept arguments"]
        )

        # %c
        expected_err = "%c requires an integer or string"
        self.assert_errors(ConversionSpecifier("c"), KnownValue(3.0), [expected_err])
        self.assert_errors(ConversionSpecifier("c"), KnownValue(3), [])
        self.assert_errors(ConversionSpecifier("c"), KnownValue("c"), [])
        self.assert_errors(
            ConversionSpecifier("c", is_bytes=True), KnownValue(b"c"), []
        )
        if six.PY3:
            self.assert_errors(
                ConversionSpecifier("c"), KnownValue(b"c"), [expected_err]
            )
            self.assert_errors(
                ConversionSpecifier("c", is_bytes=True),
                KnownValue("c"),
                ["%c on a bytes pattern requires an integer or a byte"],
            )
        else:
            self.assert_errors(ConversionSpecifier("c"), KnownValue(b"c"), [])
            self.assert_errors(
                ConversionSpecifier("c", is_bytes=True), KnownValue("c"), []
            )

        # %b
        expected_err = "%b accepts only bytes"
        self.assert_errors(ConversionSpecifier("b"), KnownValue(3), [expected_err])
        self.assert_errors(ConversionSpecifier("b"), KnownValue(b"b"), [])

        # %s
        if six.PY3:
            expected_err = "%s accepts only bytes"
            self.assert_errors(
                ConversionSpecifier("s", is_bytes=True), KnownValue("s"), [expected_err]
            )
            self.assert_errors(
                ConversionSpecifier("s", is_bytes=True), KnownValue(b"b"), []
            )
        else:
            self.assert_errors(ConversionSpecifier("s"), KnownValue("s"), [])
            self.assert_errors(
                ConversionSpecifier("s", is_bytes=True), KnownValue(b"b"), []
            )
            self.assert_errors(
                ConversionSpecifier("s"),
                KnownValue(b"b"),
                ["cannot pass bytes argument to text % string"],
            )
            self.assert_errors(
                ConversionSpecifier("s", is_bytes=True),
                KnownValue("s"),
                ["cannot pass text argument to bytes % string"],
            )
            # anything else is fine
            self.assert_errors(ConversionSpecifier("s"), KnownValue(3), [])
            self.assert_errors(
                ConversionSpecifier("s", is_bytes=True), KnownValue(3), []
            )

    def test_star_conversion_specifier(self):
        expected_err = "'*' special specifier only accepts ints"
        self.assert_errors(StarConversionSpecifier(), KnownValue("s"), [expected_err])
        if six.PY2:
            self.assert_errors(
                StarConversionSpecifier(),
                KnownValue(long(3)),  # noqlint
                [expected_err],
            )

    def test_format_string_no_specifiers(self):
        expected_err = "use of % on string with no conversion specifiers"
        self.assert_errors(PercentFormatString(""), KnownValue((1,)), [expected_err])
        self.assert_errors(PercentFormatString(""), KnownValue(3), [expected_err])
        # empty tuple or dict is accepted
        self.assert_errors(PercentFormatString(""), KnownValue(()), [])
        self.assert_errors(PercentFormatString(""), KnownValue({}), [])

    def test_format_string_dict(self):
        requires_mapping = "% string requires a mapping"

        self.assert_errors(
            PercentFormatString.from_pattern("%(a)s"), KnownValue(3), [requires_mapping]
        )
        self.assert_errors(
            PercentFormatString.from_pattern("%(a)d"),
            KnownValue({"a": "b"}),
            ["%d conversion specifier accepts numbers"],
        )
        # multiple errors for the same key get reported
        fs_aa = PercentFormatString.from_pattern("%(a)d%(a)f")
        self.assert_errors(
            fs_aa,
            KnownValue({"a": "b"}),
            [
                "%d conversion specifier accepts numbers",
                "%f conversion specifier accepts numbers",
            ],
        )
        # as do multiple errors for different keys
        fs_ac = PercentFormatString.from_pattern("%(a)d%(c)d")
        self.assert_errors(
            fs_ac,
            KnownValue({"a": "b", "c": "d"}),
            [
                "%d conversion specifier accepts numbers",
                "%d conversion specifier accepts numbers",
            ],
        )
        # but correct usage doesn't yield errors
        self.assert_errors(fs_aa, KnownValue({"a": 3}), [])
        self.assert_errors(fs_ac, KnownValue({"a": 3, "c": 3}), [])

        # DictIncompleteValue
        self.assert_errors(
            PercentFormatString.from_pattern("%(a)s"),
            DictIncompleteValue([(KnownValue("a"), KnownValue(2))]),
            [],
        )

        # missing keys are fine, since we don't know when the dict was mutated
        self.assert_errors(
            PercentFormatString.from_pattern("%(a)s"), KnownValue({}), []
        )
        # and so are extra keys
        self.assert_errors(
            PercentFormatString.from_pattern("%(a)s"), KnownValue({"a": 3, "b": 4}), []
        )

    def test_format_string_tuple(self):
        too_few = "too few arguments to format string"
        too_many = "too many arguments to format string"
        # too few
        self.assert_errors(
            PercentFormatString.from_pattern("%s%s"), KnownValue((1,)), [too_few]
        )
        self.assert_errors(
            PercentFormatString.from_pattern("%s%s"), KnownValue(1), [too_few]
        )
        self.assert_errors(
            PercentFormatString.from_pattern("%s%s"),
            SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]),
            [too_few],
        )
        self.assert_errors(
            PercentFormatString.from_pattern("%s%s"),
            SequenceIncompleteValue(tuple, [KnownValue(1)]),
            [too_few],
        )

        # too many
        self.assert_errors(
            PercentFormatString.from_pattern("%s"), KnownValue((1, 2)), [too_many]
        )
        self.assert_errors(
            PercentFormatString.from_pattern("%s"),
            SequenceIncompleteValue(tuple, [KnownValue(1), KnownValue(2)]),
            [too_many],
        )

        # just right
        self.assert_errors(
            PercentFormatString.from_pattern("%s%s"), KnownValue((1, 2)), []
        )
        self.assert_errors(PercentFormatString.from_pattern("%s"), KnownValue(1), [])
        self.assert_errors(PercentFormatString.from_pattern("%s%%"), KnownValue(1), [])
        self.assert_errors(
            PercentFormatString.from_pattern("%s"),
            SequenceIncompleteValue(list, [KnownValue(1), KnownValue(2)]),
            [],
        )
        self.assert_errors(
            PercentFormatString.from_pattern("%s"),
            SequenceIncompleteValue(tuple, [KnownValue(1)]),
            [],
        )


# Black removes some important u prefixes
# fmt: off
class TestPercentFormatString(TestNameCheckVisitorBase):
    @assert_fails(ErrorCode.bad_format_string)
    def test_too_few_values(self):
        def capybara(x):
            print('%s %s' % (x,))

    @assert_fails(ErrorCode.bad_format_string)
    def test_too_few_values_typed(self):
        def capybara(x):
            print('%s %s' % int(x))

    @assert_passes()
    def test_too_few_values_dict_typed(self):
        def capybara(x):
            print('%(capybara)s %(paca)s' % dict(x))

    @assert_passes()
    def test_bad_key_in_known_dict(self):
        def capybara():
            print('%(capybara)s' % {42: 'capybara', 'capybara': 42})

    @assert_passes()
    def test_bad_key_in_incomplete_dict(self):
        def capybara(x):
            print('%(capybara)s' % {int(x): 'capybara'})

    @assert_passes()
    def test_dict_key_is_not_format(self):
        def capybara(x):
            print('hello %s' % {'foo': x})
            print('hello %s' % {'foo': 'x'})

    @assert_fails(ErrorCode.bad_format_string)
    def test_wrong_type(self):
        def capybara(x):
            print('%d %s' % ('foo', x))

    # should pass because test_scope can't recognize if the dictionary was mutated later,
    # so we should ignore all missing keys on dictionary arguments
    @assert_passes()
    def test_missing_key(self):
        def capybara(x):
            print('%(foo)s' % {})

    @assert_passes()
    def test_none_passes(self):
        def capybara(foo):
            # to deal with some code that sets global state to None and changes it later
            print('%d %s' % (None, foo))

    @assert_fails(ErrorCode.bad_format_string)
    def test_no_format(self):
        def pacarana(foo):
            return 'dinomys' % foo

    @assert_passes()
    def test_inference(self):
        def capybara(a):
            assert_is_value('%s %s' % (3, 0), TypedValue(str))
            assert_is_value('%s %s' % (a, 0), TypedValue(str))

    @only_before((3, 0))
    @assert_fails(ErrorCode.bad_format_string)
    def test_unicode_bytes(self):
        def pacarana():
            u'foo %s' % b'bar'

    @only_before((3, 0))
    @assert_fails(ErrorCode.bad_format_string)
    def test_unicode_bytes_in_known_dict(self):
        def pacarana():
            u'foo %(bar)s' % {u'bar': b'bar'}

    @only_before((3, 0))
    @assert_fails(ErrorCode.bad_format_string)
    def test_unicode_bytes_in_incomplete_dict(self):
        def pacarana(x):
            u'foo %(bar)s' % {u'bar': bytes(x)}

    @only_before((3, 0))
    @assert_fails(ErrorCode.bad_format_string)
    def test_unicode_bytes_tuple(self):
        def pacarana():
            u'foo %s' % (b'bar',)

    @only_before((3, 0))
    @assert_fails(ErrorCode.bad_format_string)
    def test_bytes_unicode(self):
        def pacarana():
            b'foo %s' % u'bar'

    @only_before((3, 0))
    @assert_fails(ErrorCode.bad_format_string)
    def test_bytes_unicode_tuple(self):
        def pacarana():
            b'foo %s' % (u'bar',)

    @only_before((3, 0))
    @assert_fails(ErrorCode.bad_format_string)
    def test_bytes_typed(self):
        def pacarana(x):
            u'foo %s' % bytes(x)
# fmt: on


class TestUseFStrings(TestNameCheckVisitorBase):
    @skip_before((3, 6))
    def test_replacement(self):
        self.assert_is_changed(
            """
def capybara(x):
    print('capybaras %s' % x)
""",
            """
def capybara(x):
    print(f'capybaras {x}')
""",
        )
        self.assert_is_changed(
            """
def capybara(x, y):
    print('capybaras %s %s' % (x, y.z))
""",
            """
def capybara(x, y):
    print(f'capybaras {x} {y.z}')
""",
        )

    @skip_before((3, 6))
    def test_newline(self):
        self.assert_is_changed(
            r"""
def capybara(x):
    print('hello %s\n' % x)
""",
            r"""
def capybara(x):
    print(f'hello {x}\n')
""",
        )

    @skip_before((3, 6))
    @assert_passes()
    def test_bytes(self):
        def capybara(x):
            b"foo %s\n" % x

    @skip_before((3, 6))
    @assert_passes()
    def test_braces(self):
        def capybara(x):
            "foo {%s}" % x

    @skip_before((3, 6))
    @assert_passes()
    def test_fancy_conversions(self):
        def capybara(x):
            "foo %.3s" % x

    @skip_before((3, 6))
    @assert_passes()
    def test_mapping(self):
        def capybara(x):
            "foo %(x)s" % {"x": x}

    @skip_before((3, 6))
    @assert_passes()
    def test_conversion_type(self):
        def capybara(x):
            "foo %f" % x

    @skip_before((3, 6))
    @assert_passes()
    def test_complicated_expression(self):
        def capybara(x):
            "foo %s" % len(x)
