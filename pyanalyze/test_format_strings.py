# static analysis: ignore
from .error_code import ErrorCode
from .format_strings import (
    ConversionSpecifier,
    PercentFormatString,
    StarConversionSpecifier,
    FormatString,
    ReplacementField,
    IndexOrAttribute,
    parse_format_string,
)
from .value import (
    KVPair,
    assert_is_value,
    AnySource,
    AnyValue,
    KnownValue,
    DictIncompleteValue,
    SequenceIncompleteValue,
    TypedValue,
)
from .test_node_visitor import assert_passes
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_value import CTX


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
        is_bytes = isinstance(pattern, bytes)
        if is_bytes:
            expected = PercentFormatString.from_bytes_pattern(pattern)
        else:
            expected = PercentFormatString.from_pattern(pattern)
        assert (
            PercentFormatString(
                pattern, is_bytes, specifiers=specifiers, raw_pieces=raw_pieces
            )
            == expected
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
    ("{a:3}", [ReplacementField("a", format_spec=FormatString(["3"]))]),
    (
        "{a:{b}}",
        [ReplacementField("a", format_spec=FormatString([ReplacementField("b")]))],
    ),
    (
        "{a.b!r:{c:{d}}}",
        [
            ReplacementField(
                "a",
                index_attribute=((IndexOrAttribute.attribute, "b"),),
                conversion="r",
                format_spec=FormatString(
                    [
                        ReplacementField(
                            "c", format_spec=FormatString([ReplacementField("d")])
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
        assert (FormatString(expected), []) == parse_format_string(
            format_string
        ), format_string
    for format_string, position, message in DOT_FORMAT_ERRORS:
        _, errors = parse_format_string(format_string)
        assert len(errors) >= 1
        assert (position, message) == errors[0], format_string


def assert_lints(pattern, errors):
    """Asserts that linting this pattern produces the given errors."""
    fs = PercentFormatString.from_pattern(pattern)
    assert errors == list(fs.lint()), f"while linting {pattern}"


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


class TestAccept(object):
    def assert_errors(self, obj, arg, expected):
        actual = list(obj.accept(arg, CTX))
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
        self.assert_errors(ConversionSpecifier("d"), AnyValue(AnySource.marker), [])
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
        expected_err = "%c requires an integer or character"
        self.assert_errors(ConversionSpecifier("c"), KnownValue(3.0), [expected_err])
        self.assert_errors(ConversionSpecifier("c"), KnownValue(3), [])
        self.assert_errors(ConversionSpecifier("c"), KnownValue("c"), [])
        self.assert_errors(
            ConversionSpecifier("c", is_bytes=True), KnownValue(b"c"), []
        )
        self.assert_errors(ConversionSpecifier("c"), KnownValue(b"c"), [expected_err])
        self.assert_errors(
            ConversionSpecifier("c", is_bytes=True),
            KnownValue("c"),
            ["%c requires an integer or character"],
        )

        # %b
        expected_err = "%b accepts only bytes"
        self.assert_errors(ConversionSpecifier("b"), KnownValue(3), [expected_err])
        self.assert_errors(ConversionSpecifier("b"), KnownValue(b"b"), [])

        # %s
        expected_err = "%s accepts only bytes"
        self.assert_errors(
            ConversionSpecifier("s", is_bytes=True), KnownValue("s"), [expected_err]
        )
        self.assert_errors(
            ConversionSpecifier("s", is_bytes=True), KnownValue(b"b"), []
        )

    def test_star_conversion_specifier(self):
        expected_err = "'*' special specifier only accepts ints"
        self.assert_errors(StarConversionSpecifier(), KnownValue("s"), [expected_err])

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
            DictIncompleteValue(dict, [KVPair(KnownValue("a"), KnownValue(2))]),
            [],
        )

        # missing keys are an error
        self.assert_errors(
            PercentFormatString.from_pattern("%(a)s"),
            KnownValue({}),
            ["No value specified for keys a"],
        )
        # but extra keys are fine
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


class TestPercentFormatString(TestNameCheckVisitorBase):
    @assert_passes(settings={ErrorCode.use_fstrings: False})
    def test_basic(self):
        def capybara(x):
            print("%s %s" % (x,))  # E: bad_format_string
            print("%s %s" % int(x))  # E: bad_format_string
            print("%(capybara)s %(paca)s" % dict(x))
            # extra key is fine
            print("%(capybara)s" % {42: "capybara", "capybara": 42})
            print("%(capybara)s" % {int(x): "capybara"})
            # if we're not using dict formatting, passing a dict is still fine
            print("hello %s" % {"foo": x})
            print("hello %s" % {"foo": "x"})
            print("%d %s" % ("foo", x))  # E: bad_format_string
            print("%(foo)s" % {})  # E: bad_format_string
            print("%d %s" % (None, x))  # E: bad_format_string
            print("dinomys" % x)  # E: bad_format_string

    @assert_passes(settings={ErrorCode.use_fstrings: False})
    def test_mvv(self):
        from typing import Union

        def capybara(x: Union[int, float]):
            print("%f" % (x,))
            print("%*d" % (x, x))  # E: bad_format_string

    @assert_passes(settings={ErrorCode.use_fstrings: False})
    def test_character(self):
        def capybara(i: int, s: str, b: bytes):
            print("%c" % i)
            print("%c" % s)
            print("%c" % b)  # E: bad_format_string
            print("%c" % 42)
            print("%c" % 257)  # E: bad_format_string
            print("%c" % -1)  # E: bad_format_string
            print("%c" % "x")
            print("%c" % "ab")  # E: bad_format_string

            print(b"%c" % i)
            print(b"%c" % s)  # E: bad_format_string
            print(b"%c" % b)

    @assert_passes()
    def test_inference(self):
        def capybara(a):
            assert_is_value("%s %s" % (3, 0), TypedValue(str))
            assert_is_value("%s %s" % (a, 0), TypedValue(str))


class TestUseFStrings(TestNameCheckVisitorBase):
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

    @assert_passes()
    def test_bytes(self):
        def capybara(x):
            b"foo %s\n" % x

    @assert_passes()
    def test_braces(self):
        def capybara(x):
            "foo {%s}" % x

    @assert_passes()
    def test_fancy_conversions(self):
        def capybara(x):
            "foo %.3s" % x

    @assert_passes()
    def test_mapping(self):
        def capybara(x):
            "foo %(x)s" % {"x": x}

    @assert_passes()
    def test_conversion_type(self):
        def capybara(x):
            "foo %f" % x

    @assert_passes()
    def test_complicated_expression(self):
        def capybara(x):
            "foo %s" % len(x)
