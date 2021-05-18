"""

Module for checking %-formatted and .format()-formatted strings.

"""

import ast
from collections import defaultdict
from dataclasses import dataclass, field
import enum
import numbers
import re
import sys
from collections.abc import Mapping
from typing import (
    Callable,
    Dict,
    Iterable,
    Match,
    Optional,
    Sequence,
    Union,
    List,
    Tuple,
)
from typing_extensions import Literal

from .error_code import ErrorCode
from .value import (
    AnnotatedValue,
    KnownValue,
    DictIncompleteValue,
    SequenceIncompleteValue,
    TypedValue,
    MultiValuedValue,
    UNRESOLVED_VALUE,
    Value,
    VariableNameValue,
)

# refer to https://docs.python.org/2/library/stdtypes.html#string-formatting-operations
_FORMAT_STRING_REGEX = r"""
    (?P<pre_match>.*?)  # stuff before the match
    (
        %  # starting character
        (?P<mapping_key>\([^\)]+\))?
        (?P<conversion_flags>[#0\- +]+)?
        (?P<field_width>\*|\d+)?
        (?P<precision>\.(\*|\d+))?
        (?P<length_modifier>[hlL])?
        (?P<conversion_type>[diouxXeEfFgGcrs%ba])
    |
        $  # or until the end of the string
    )
"""
_FLAGS = re.VERBOSE | re.DOTALL
_FORMAT_STRING_REGEX_BYTES = re.compile(_FORMAT_STRING_REGEX.encode("ascii"), _FLAGS)
_FORMAT_STRING_REGEX_TEXT = re.compile(_FORMAT_STRING_REGEX, _FLAGS)
# All conversion types that accept numeric arguments
_NUMERIC_CONVERSION_TYPES = set("diouxXeEfFgG")
_FORMAT_STRING_CONVERSIONS = {"r", "s", "a"}
_IDENTIFIER_REGEX = re.compile(r"^[A-Za-z_][A-Za-z_\d]*$")

#
# % formatting
#


@dataclass
class ConversionSpecifier:
    """Class representing a single conversion specifier in a format string."""

    conversion_type: str
    mapping_key: Optional[str] = None
    conversion_flags: Optional[str] = None
    field_width: Union[int, Literal["*"], None] = None
    precision: Union[int, Literal["*"], None] = None
    length_modifier: Optional[str] = None
    is_bytes: bool = False

    @classmethod
    def from_match(
        cls, match: Union[Match[bytes], Match[str]]
    ) -> "ConversionSpecifier":
        """Argument is a match returned by _FORMAT_STRING_REGEX."""
        conversion_type = match.group("conversion_type")
        is_bytes = isinstance(conversion_type, bytes)
        mapping_key = match.group("mapping_key")
        if mapping_key is not None:
            mapping_key = mapping_key[1:-1]  # remove enclosing parens
        field_width = match.group("field_width")
        if field_width is not None:
            field_width = cls._parse_int_field(field_width)
        precision = match.group("precision")
        if precision is not None:
            precision = cls._parse_int_field(precision[1:])
        return cls(
            conversion_type=cls._maybe_decode(conversion_type),
            mapping_key=cls._maybe_decode(mapping_key)
            if mapping_key is not None
            else None,
            conversion_flags=cls._maybe_decode(match.group("conversion_flags")),
            field_width=field_width,
            precision=precision,
            length_modifier=cls._maybe_decode(match.group("length_modifier")),
            is_bytes=is_bytes,
        )

    @classmethod
    def _maybe_decode(cls, string: Union[str, bytes]) -> str:
        """We want to treat all fields as text even on a bytes pattern for simplicity."""
        if isinstance(string, bytes):
            return string.decode("ascii")
        else:
            return string

    @classmethod
    def _parse_int_field(cls, raw: Union[str, bytes]) -> Union[int, Literal["*"]]:
        """Helper for parsing match results for the field_width and precision fields."""
        if isinstance(raw, bytes):
            raw = raw.decode("ascii")
        if raw == "*":
            return "*"
        else:
            return int(raw)

    def lint(self) -> Iterable[str]:
        """Finds any errors in this specifier."""
        if self.conversion_type == "%":
            if any(
                f is not None
                for f in (
                    self.mapping_key,
                    self.conversion_flags,
                    self.field_width,
                    self.precision,
                    self.length_modifier,
                )
            ):
                # here you're using %% escaping, but you use one of the optional specifiers, none
                # of which will do anything
                yield "using % combined with optional specifiers does not make sense"
        elif self.conversion_type == "b":
            if not self.is_bytes:
                yield (
                    "the %b conversion specifier works only on Python 3 bytes patterns"
                )

    def accept(self, arg: Value) -> Iterable[str]:
        """Produces any errors from passing the given object to this specifier."""
        if isinstance(arg, AnnotatedValue):
            arg = arg.value
        if arg is UNRESOLVED_VALUE or isinstance(arg, MultiValuedValue):
            return
        if self.conversion_type in _NUMERIC_CONVERSION_TYPES:
            # to deal with some code that sets global state to None and changes it later
            if (
                not arg.is_type(numbers.Number)
                and arg != KnownValue(None)
                and not isinstance(arg, VariableNameValue)
            ):
                yield "%{} conversion specifier accepts numbers, not {!r}".format(
                    self.conversion_type, arg
                )
        elif self.conversion_type in ("a", "r"):
            # accepts anything
            pass
        elif self.conversion_type == "c":
            if arg.is_type(int) or arg.is_type(str):
                if self.is_bytes and arg.is_type(str):
                    yield (
                        "%c on a bytes pattern requires an integer or a byte, not {!r}"
                        .format(arg)
                    )
            elif self.is_bytes and arg.is_type(bytes):
                # in Python 3, b'%c' % b'c' works but not '%c' % b'c'
                pass
            else:
                yield "%c requires an integer or string, not {!r}".format(arg)
        elif self.conversion_type == "b" or (
            self.is_bytes and self.conversion_type == "s"
        ):
            # in Python 3 bytes patterns, s is equivalent to b
            if not arg.is_type(bytes):
                yield "%{} accepts only bytes, not {}".format(self.conversion_type, arg)
        elif self.conversion_type == "s":
            # accepts anything
            pass
        elif self.conversion_type == "%":
            yield "%% does not accept arguments"
        else:
            # should never happen
            assert False, "unhandled conversion type {}".format(self.conversion_type)


class StarConversionSpecifier:
    """Fake conversion specifier for the '*' special cases for field width and precision."""

    def accept(self, arg: Value) -> Iterable[str]:
        # it doesn't accept longs in Python 2
        if arg is not UNRESOLVED_VALUE and not arg.is_type(int):
            yield "'*' special specifier only accepts ints, not {}".format(arg)


@dataclass
class PercentFormatString:
    """Class representing a parsed % format string.

    pattern is the original string
    is_bytes is whether the pattern is bytes or text
    specifiers is a sequence of ConversionSpecifiers
    raw_pieces are the string pieces before, between, and after the specifiers

    """

    pattern: Union[bytes, str]
    is_bytes: bool = False
    specifiers: Sequence[ConversionSpecifier] = ()
    raw_pieces: Sequence[str] = ()

    @classmethod
    def from_pattern(cls, pattern: str) -> "PercentFormatString":
        """Creates a parsed PercentFormatString from a raw string."""
        if not isinstance(pattern, str):
            raise TypeError("invalid type for format string: {!r}".format(pattern))
        matches = list(_FORMAT_STRING_REGEX_TEXT.finditer(pattern))
        specifiers = tuple(
            ConversionSpecifier.from_match(match)
            for match in matches
            if match.group("conversion_type") is not None
        )
        raw_pieces = [match.group("pre_match") for match in matches]
        if len(raw_pieces) == len(specifiers) + 2:
            raw_pieces = raw_pieces[:-1]
        if pattern.endswith("\n"):
            # due to a quirk in the re module, the final newline otherwise gets removed
            raw_pieces[-1] += "\n"
        return cls(
            pattern, is_bytes=False, specifiers=specifiers, raw_pieces=tuple(raw_pieces)
        )

    @classmethod
    def from_bytes_pattern(cls, pattern: bytes) -> "PercentFormatString":
        """Creates a parsed PercentFormatString from a raw bytestring."""
        matches = list(_FORMAT_STRING_REGEX_BYTES.finditer(pattern))
        specifiers = tuple(
            ConversionSpecifier.from_match(match)
            for match in matches
            if match.group("conversion_type") is not None
        )
        raw_pieces = [match.group("pre_match") for match in matches]
        if len(raw_pieces) == len(specifiers) + 2:
            raw_pieces = raw_pieces[:-1]
        if pattern.endswith(b"\n"):
            # due to a quirk in the re module, the final newline otherwise gets removed
            raw_pieces[-1] += b"\n"
        return cls(
            pattern, is_bytes=True, specifiers=specifiers, raw_pieces=tuple(raw_pieces)
        )

    def needs_mapping(self) -> bool:
        """Returns whether this format string requires a mapping as an argument."""
        return any(cs.mapping_key is not None for cs in self.specifiers)

    def lint(self) -> Iterable[str]:
        """Finds errors in the pattern itself."""
        needs_mapping = self.needs_mapping()
        for cs in self.specifiers:
            for err in cs.lint():
                yield err
            if needs_mapping:
                if (
                    cs.mapping_key is None
                    or cs.precision == "*"
                    or cs.field_width == "*"
                ):
                    yield (
                        "cannot combine specifiers that require a mapping with those"
                        " that do not"
                    )
        for piece in self.raw_pieces:
            if (b"%" in piece) if self.is_bytes else ("%" in piece):
                yield "invalid conversion specifier in {}".format(piece)

    def accept(self, args: Value) -> Iterable[str]:
        """Checks whether this format string can accept the given Value as arguments."""
        if not self.specifiers:
            # if there are no conversion specifiers and we're doing % formatting anyway, throw an
            # error
            # this will produce errors for some things that aren't errors at runtime, but these seem
            # unlikely to appear in legitimate code (things like "'' % {'a': 3}")
            # but if the args are known to be an empty tuple or dict, ignore it
            if args != KnownValue(()) and args != KnownValue({}):
                yield "use of % on string with no conversion specifiers"
        elif self.needs_mapping():
            yield from self.accept_mapping_args(args)
        else:
            yield from self.accept_tuple_args(args)

    def get_specifier_mapping(self) -> Dict[str, List[ConversionSpecifier]]:
        """Return a mapping from mapping key to conversion specifiers for that mapping key."""
        out = defaultdict(list)
        for specifier in self.specifiers:
            if specifier.conversion_type != "%":
                out[specifier.mapping_key].append(specifier)
        return out

    def accept_mapping_args(self, args: Value) -> Iterable[str]:
        cs_map = self.get_specifier_mapping()
        if isinstance(args, KnownValue):
            if isinstance(args.val, Mapping):
                for key, value in args.val.items():
                    for specifier in cs_map[key]:
                        yield from specifier.accept(KnownValue(value))
            else:
                yield "% string requires a mapping, not {}".format(args)
        elif isinstance(args, DictIncompleteValue):
            for key, value in args.items:
                if isinstance(key, KnownValue):
                    for specifier in cs_map[key.val]:
                        yield from specifier.accept(value)
        elif args is not UNRESOLVED_VALUE and not args.is_type(Mapping):
            yield "% string requires a mapping, not {}".format(args)

    def get_serial_specifiers(
        self,
    ) -> Iterable[Union[ConversionSpecifier, StarConversionSpecifier]]:
        """Returns all specifiers to use when formatting with a tuple."""
        for specifier in self.specifiers:
            if specifier.field_width == "*":
                yield StarConversionSpecifier()
            if specifier.precision == "*":
                yield StarConversionSpecifier()
            if specifier.conversion_type != "%":
                yield specifier

    def accept_tuple_args(self, args: Value) -> Iterable[str]:
        specifiers = list(self.get_serial_specifiers())
        if isinstance(args, AnnotatedValue):
            args = args.value
        if args.is_type(tuple):
            if isinstance(args, SequenceIncompleteValue):
                all_args = args.members
            elif isinstance(args, KnownValue):
                all_args = tuple(KnownValue(elt) for elt in args.val)
            else:
                # it's a tuple but we don't know what's in it, so assume it's ok
                return
        elif args is UNRESOLVED_VALUE:
            return
        elif isinstance(args, MultiValuedValue):
            if any(v is UNRESOLVED_VALUE or v.is_type(tuple) for v in args.vals):
                return
            else:
                all_args = (args,)
        else:
            all_args = (args,)
        num_args = len(all_args)
        num_specifiers = len(specifiers)
        if num_args < num_specifiers:
            yield "too few arguments to format string: got {} but expected {}".format(
                num_args, num_specifiers
            )
        elif num_args > num_specifiers:
            yield "too many arguments to format string: got {} but expected {}".format(
                num_args, num_specifiers
            )
        else:
            for arg, specifier in zip(all_args, specifiers):
                yield from specifier.accept(arg)


def check_string_format(
    node: ast.AST,
    format_str: Union[str, bytes],
    args_node: ast.AST,
    args: Value,
    on_error: Callable[..., None],
) -> Tuple[Value, Optional[ast.AST]]:
    """Checks that arguments to %-formatted strings are correct."""
    if isinstance(format_str, bytes):
        fs = PercentFormatString.from_bytes_pattern(format_str)
    else:
        fs = PercentFormatString.from_pattern(format_str)
    for err in fs.lint():
        on_error(node, err, error_code=ErrorCode.bad_format_string)
    for err in fs.accept(args):
        on_error(node, err, error_code=ErrorCode.bad_format_string)
    return TypedValue(type(format_str)), maybe_replace_with_fstring(fs, args_node)


def maybe_replace_with_fstring(
    fs: PercentFormatString, args_node: ast.AST
) -> Optional[ast.AST]:
    """If appropriate, emits an error to replace this % format with an f-string."""
    # otherwise there are no f-strings
    if sys.version_info < (3, 6):
        return None
    # there are no bytes f-strings
    if isinstance(fs.pattern, bytes):
        return None
    # if there is a { in the string, we will need escaping in order to use an f-string, which might
    # make the code worse
    if any("{" in piece or "}" in piece for piece in fs.raw_pieces):
        return None
    # special conversion specifiers are rare and more difficult to replace, so just ignore them for
    # now
    if any(
        any(
            [
                cs.mapping_key,
                cs.conversion_flags,
                cs.field_width,
                cs.precision,
                cs.length_modifier,
            ]
        )
        for cs in fs.specifiers
    ):
        return None
    # don't attempt fancy conversion types
    if any(cs.conversion_type not in ("d", "s") for cs in fs.specifiers):
        return None
    # only proceed if all the arguments are simple (currently, names or attribute accesses)
    if isinstance(args_node, ast.Tuple):
        if any(not _is_simple_enough(elt) for elt in args_node.elts):
            return None
        substitutions = args_node.elts
    elif len(fs.specifiers) == 1:
        if not _is_simple_enough(args_node):
            return None
        substitutions = [args_node]
    else:
        return None
    # the linter should have given an error in this case
    if len(substitutions) != len(fs.specifiers) != len(fs.raw_pieces) - 1:
        return None
    parts = []
    for raw_piece, substitution in zip(fs.raw_pieces, substitutions):
        if raw_piece:
            parts.append(ast.Str(s=raw_piece))
        parts.append(
            ast.FormattedValue(value=substitution, conversion=-1, format_spec=None)
        )
    if fs.raw_pieces[-1]:
        parts.append(ast.Str(s=fs.raw_pieces[-1]))
    return ast.JoinedStr(values=parts)


def _is_simple_enough(node: ast.AST) -> bool:
    """Returns whether a node is simple enough to be substituted into an f-string."""
    if isinstance(node, ast.Name):
        return True
    elif isinstance(node, ast.Attribute):
        return _is_simple_enough(node.value)
    else:
        return False


#
# .format()
#

FormatErrors = List[Tuple[int, str]]
Children = List[Union[str, "ReplacementField"]]


@dataclass
class _ParserState:
    string: str
    current_index: int = 0
    errors: FormatErrors = field(default_factory=list)

    def peek(self) -> Optional[str]:
        if self.current_index >= len(self.string):
            return None
        return self.string[self.current_index]

    def next(self) -> Optional[str]:
        char = self.peek()
        self.current_index += 1
        return char

    def add_error(self, message: str) -> None:
        self.errors.append((self.current_index, message))


@dataclass
class FormatString:
    children: Children

    def iter_replacement_fields(self) -> Iterable["ReplacementField"]:
        """Iterator over all child replacement fields."""
        for child in self.children:
            if isinstance(child, ReplacementField):
                yield from child.iter_replacement_fields()


class IndexOrAttribute(enum.Enum):
    """Either a [index] or .attribute.

    The syntax for a field is:

        field_name        ::=  arg_name ("." attribute_name | "[" element_index "]")*

    Therefore, a template like '{a.b[0].c}' is allowed. We represent this by setting
    the index_attribute attribute of the ReplacementField to:

        [
            (IndexOrAttribute.attribute, "b"),
            (IndexOrAttribute.index, "0"),
            (IndexOrAttribute.attribute, "c")
        ]

    """

    index = 1
    attribute = 2


@dataclass
class ReplacementField:
    arg_name: Union[None, int, str]
    index_attribute: Sequence[Tuple[IndexOrAttribute, str]] = ()
    conversion: Optional[str] = None
    format_spec: Optional[FormatString] = None

    def iter_replacement_fields(self) -> Iterable["ReplacementField"]:
        """Iterator over all child replacement fields."""
        yield self
        if self.format_spec:
            for child in self.format_spec.children:
                if isinstance(child, ReplacementField):
                    yield from child.iter_replacement_fields()


def parse_format_string(string: str) -> Tuple[FormatString, FormatErrors]:
    state = _ParserState(string)
    children = _parse_children(state, end_at=None)
    return FormatString(children), state.errors


def _parse_children(state: _ParserState, end_at: Optional[str] = None) -> Children:
    children = []
    current_literal = []
    while True:
        char = state.next()
        if char is None:
            if end_at is None:
                if current_literal:
                    children.append("".join(current_literal))
            else:
                state.add_error("expected '{}' before end of string".format(end_at))
            break
        elif char == end_at:
            if current_literal:
                children.append("".join(current_literal))
            break
        elif char == "{":
            next_char = state.peek()
            if next_char == "{":
                state.next()
                current_literal.append("{")
            else:
                if current_literal:
                    children.append("".join(current_literal))
                current_literal = []
                children.append(_parse_replacement_field(state))
        elif char == "}":
            next_char = state.peek()
            if next_char == "}":
                state.next()
                current_literal.append("}")
            else:
                state.add_error("single '}' encountered in format string")
        else:
            current_literal.append(char)
    return children


def _parse_replacement_field(state: _ParserState) -> Union[str, ReplacementField]:
    arg_name_chars = []
    index_attribute = []
    conversion = None
    format_spec = None
    specials = {"}", ".", "[", "!", ":"}
    allowed_specials = specials
    while True:
        char = state.next()
        if char is None:
            state.add_error("expected '}' before end of string")
            return ""
        elif char in specials:
            if char not in allowed_specials:
                state.add_error(
                    "expected one of {}".format(
                        ", ".join("'{}'".format(c) for c in sorted(allowed_specials))
                    )
                )
                return ""
            elif char == "}":
                break
            elif char == ".":
                attribute_chars = []
                while True:
                    char = state.peek()
                    if char is None:
                        state.add_error("expected '}' before end of string")
                        return ""
                    elif char in specials:
                        break
                    else:
                        state.next()
                        attribute_chars.append(char)
                attribute_name = "".join(attribute_chars)
                if not _IDENTIFIER_REGEX.match(attribute_name):
                    state.add_error("invalid attribute '{}'".format(attribute_name))
                    return ""
                index_attribute.append((IndexOrAttribute.attribute, attribute_name))
            elif char == "[":
                index_string = []
                while True:
                    char = state.next()
                    if char is None:
                        state.add_error("expected ']' before end of string")
                        return ""
                    elif char == "]":
                        break
                    else:
                        index_string.append(char)
                index_attribute.append((IndexOrAttribute.index, "".join(index_string)))
            elif char == "!":
                conversion = state.next()
                allowed_specials = {":", "}"}
                if conversion not in _FORMAT_STRING_CONVERSIONS:
                    state.add_error(
                        "Unknown conversion specifier '{!s}'".format(conversion)
                    )
                    return ""
                next_char = state.peek()
                if next_char not in allowed_specials:
                    state.add_error(
                        "expected one of {}".format(
                            ", ".join(
                                "'{}'".format(c) for c in sorted(allowed_specials)
                            )
                        )
                    )
                    return ""
            elif char == ":":
                format_spec = FormatString(_parse_children(state, "}"))
                break
        elif char == "{":
            state.add_error("unexpected '{' in field name")
            return ""
        else:
            arg_name_chars.append(char)
    arg_name_str = "".join(arg_name_chars)
    if not arg_name_str:
        arg_name = None
    elif arg_name_str.isdigit():
        arg_name = int(arg_name_str)
    else:
        arg_name = arg_name_str
    return ReplacementField(arg_name, tuple(index_attribute), conversion, format_spec)
