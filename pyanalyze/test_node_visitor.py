# static analysis: ignore
import ast
from ast_decompiler import decompile
from collections import defaultdict
import difflib
import enum
import functools
import inspect
import itertools
import re
import sys
import textwrap

from .node_visitor import (
    BaseNodeVisitor,
    VisitorError,
    Replacement,
    NodeTransformer,
    ReplaceNodeTransformer,
    ReplacingNodeVisitor,
)


# Base class for other tests
class BaseNodeVisitorTester(object):
    visitor_cls = None

    def _run_str(self, code_str, expect_failure=False, apply_changes=False, **kwargs):
        """Runs code_str with this visitor."""
        kwargs.setdefault("fail_after_first", not apply_changes)
        # Can be bytes in Python 2.
        if isinstance(code_str, bytes):
            code_str = code_str.decode("utf-8")
        tree = ast.parse(code_str, "<test input>")
        try:
            result = self._run_tree(
                code_str, tree, apply_changes=apply_changes, **kwargs
            )
        except VisitorError as e:
            if expect_failure:
                return e
            else:
                raise
        else:
            if expect_failure:
                assert False, 'Expected check of "%s" to fail' % code_str
        return result

    def assert_passes(self, code_str, **kwargs):
        """Asserts that running the given code_str throws no errors."""
        code_str = textwrap.dedent(code_str)
        errors = self._run_str(
            code_str, expect_failure=False, fail_after_first=False, **kwargs
        )
        expected_errors = defaultdict(lambda: defaultdict(int))
        for i, line in enumerate(code_str.splitlines(), start=1):
            whole_line_match = re.match(r"^ *#\s*E:\s*([a-z_]+)$", line)
            if whole_line_match:
                expected_errors[i + 1][whole_line_match.group(1)] += 1
                continue
            for separate_match in re.finditer(r"#\s*E:\s*([a-z_]+)", line):
                expected_errors[i][separate_match.group(1)] += 1

        mismatches = []

        for error in errors:
            lineno = error["lineno"]
            actual_code = error["code"].name
            if (
                actual_code in expected_errors[lineno]
                and expected_errors[lineno][actual_code] > 0
            ):
                expected_errors[lineno][actual_code] -= 1
            else:
                mismatches.append(
                    f"Did not expect error {actual_code} on line {lineno}"
                )

        for lineno, errors_by_line in expected_errors.items():
            for error_code, count in errors_by_line.items():
                if count > 0:
                    mismatches.append(f"Expected {error_code} on line {lineno}")

        assert not mismatches, "".join(line + "\n" for line in mismatches) + "".join(
            error["message"] for error in errors
        )

    def assert_fails(self, expected_error_code, code_str, **kwargs):
        """Asserts that running the given code_str fails with expected_error_code."""
        exc = self._run_str(code_str, expect_failure=True, **kwargs)
        assert (
            expected_error_code == exc.error_code
        ), f"{exc} does not have code {expected_error_code}"

    def assert_is_changed(self, code_str, expected_code_str, repeat=False, **kwargs):
        """Asserts that the given code_str is corrected by the visitor to expected_code_str.

        If repeat is True, repeatedly applies the visitor until the code no longer changes.

        """
        if repeat:
            while True:
                output = self._run_str(code_str, apply_changes=True, **kwargs)[1]
                if output == code_str:
                    break
                else:
                    code_str = output
        else:
            output = self._run_str(code_str, apply_changes=True, **kwargs)[1]
        assert_code_equal(expected_code_str, output)

    def _run_tree(self, code_str, tree, apply_changes=False, **kwargs):
        """Runs the visitor on this tree."""
        return self.visitor_cls(
            "<test input>", code_str, tree, **kwargs
        ).check_for_test(apply_changes=apply_changes)


def assert_passes(**kwargs):
    """Decorator for test cases that assert that a code block contains no errors.

    The body of the decorated function is executed by the NodeVisitor. If the visitor finds any
    error, the test fails.

    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self):
            self.assert_passes(_extract_code_from_fn(fn), **kwargs)

        return wrapper

    return decorator


def assert_fails(expected_error_code, **kwargs):
    """Decorator for test cases that assert that the visitor finds a specific error.

    The body of the decorated function is executed by the NodeVisitor. If the visitor does not
    produce expected_error_code, the test fails.

    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self):
            self.assert_fails(expected_error_code, _extract_code_from_fn(fn), **kwargs)

        return wrapper

    return decorator


def _extract_code_from_fn(fn):
    lines, starting_lineno = inspect.getsourcelines(fn)
    lines = [
        line.decode("utf-8") if isinstance(line, bytes) else line for line in lines
    ]
    # remove common indentation in all lines
    lines = [line + "\n" for line in textwrap.dedent("".join(lines)).splitlines()]
    # remove the def line and the decorator
    lines = reversed(
        list(itertools.takewhile(lambda l: re.match(r"^\s", l), reversed(lines)))
    )
    return textwrap.dedent("".join(lines))


# Internal tests
class ErrorCode(enum.Enum):
    no_strings = 1
    no_while = 2
    no_dicts = 3


class VeryStrictVisitor(BaseNodeVisitor):
    def visit_Str(self, node):
        self.show_error(
            node, "Strings are not allowed", error_code=ErrorCode.no_strings
        )

    def visit_Dict(self, node):
        self.show_error(
            node,
            "Dicts are not allowed",
            error_code=ErrorCode.no_dicts,
            ignore_comment="# dict: ignore",
        )


class TestVeryStrictVisitor(BaseNodeVisitorTester):
    visitor_cls = VeryStrictVisitor

    @assert_fails(ErrorCode.no_strings)
    def test_fails(self):
        "string"

    @assert_passes()
    def test_succeeds(self):
        no(strings)

    def test_multiline_string_success(self):
        code_string = """
a = 1 + 2
"""
        # should not throw an error
        self.assert_passes(code_string)

        code_string = """
a = 'foo' + 'bar'
"""
        self.assert_fails(ErrorCode.no_strings, code_string)

    def test_lineno(self):
        code_string = """# line 1
# line 2
# line 3
# line 4
h.translate('{foo')  # line 5
# line 6
# line 7
# line 8
# line 9
"""
        try:
            self._run_str(code_string)
        except VisitorError as e:
            assert "   1:" not in str(e)
            for lineno in range(2, 9):
                assert "   %d:" % lineno in str(e)
                assert "# line %d" % lineno in str(e)
            # should be outside the three context lines
            for lineno in (1, 9):
                assert "   %d:" % lineno not in str(e)
                assert "# line %d" % lineno not in str(e)
        else:
            assert False, "Expected a parse error"

    @assert_passes()
    def test_ignore_error_same_line(self):
        "string"  # static analysis: ignore
        "another string"  # static analysis: ignore[no_strings]

    @assert_passes()
    def test_ignore_error_new_line(self):
        # static analysis: ignore
        print("string")
        # static analysis: ignore[no_strings]
        print("string")

    @assert_fails(ErrorCode.no_dicts)
    def test_no_dicts(self):
        # make sure a different error code does not get picked up
        {}  # static analysis: ignore[no_strings]

    @assert_passes()
    def test_custom_ignore_comment(self):
        {}  # dict: ignore


class DuplicateVisitor(BaseNodeVisitor):
    def visit_While(self, node):
        self.show_error(node, "while loops take too long", ErrorCode.no_while)
        self.show_error(node, "while loops take too long", ErrorCode.no_while)


class TestDuplicateVisitor(BaseNodeVisitorTester):
    visitor_cls = DuplicateVisitor

    def test_no_duplicate(self):
        errors = self._run_str("while True: pass", fail_after_first=False)
        assert len(errors) == 1, errors


class NoWhileVisitor(BaseNodeVisitor):
    def visit_While(self, node):
        replacement = Replacement(
            [node.lineno], [self._lines()[node.lineno - 1].replace("while", "if")]
        )
        self.show_error(
            node,
            "while loops take too long",
            ErrorCode.no_while,
            replacement=replacement,
        )
        self.generic_visit(node)


class TestNoWhileVisitor(BaseNodeVisitorTester):
    visitor_cls = NoWhileVisitor

    def test(self):
        self.assert_is_changed(
            """
while True:
    print('capybaras!')
""",
            """
if True:
    print('capybaras!')
""",
        )


class ExampleNodeTransformer(NodeTransformer):
    def visit_Assert(self, node):
        node = self.generic_visit(node)
        return [node, node]

    def visit_Raise(self, node):
        return None

    def visit_BinOp(self, node):
        return ast.BinOp(self.visit(node.left), ast.RShift(), self.visit(node.right))


class TestNodeTransformer(object):
    transformer_cls = ExampleNodeTransformer

    def assert_is_changed(self, old_code, new_code):
        old_ast = ast.parse(old_code)
        new_ast = self.transformer_cls().visit(old_ast)
        assert_code_equal(new_code, decompile(new_ast))

    def test_returns_list(self):
        self.assert_is_changed("assert False\n", "assert False\nassert False\n")

    def test_returns_none(self):
        self.assert_is_changed("x = 3\nraise x\n", "x = 3\n")

    def test_is_changed(self):
        self.assert_is_changed("x + 3\n", "x >> 3\n")


class TestReplaceNodeTransformer(object):
    def test_found(self):
        node = ast.parse("a.b(c)")
        replacement_node = ast.Name(id="d")
        new_node = ReplaceNodeTransformer(
            node.body[0].value.func, replacement_node
        ).visit(node)
        # ensure it doesn't mutate the existing node in place
        assert new_node is not node
        assert_code_equal("d(c)\n", decompile(new_node))

    def test_not_found(self):
        node = ast.parse("a.b(c)")
        random_node = ast.Name(id="d")
        new_node = ReplaceNodeTransformer(random_node, node.body[0].value.func).visit(
            node
        )
        assert new_node is not node
        assert_code_equal("a.b(c)\n", decompile(new_node))


class HouseDivided(ReplacingNodeVisitor):
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Div):
            new_node = ast.BinOp(left=node.left, op=ast.Mult(), right=node.right)
            self.show_error(
                node,
                "A house divided cannot stand",
                replacement=self.replace_node(node, new_node),
            )
        elif isinstance(node.op, ast.Mult):
            new_node = ast.BinOp(left=node.left, op=ast.Pow(), right=node.right)
            self.show_error(
                node,
                "Go forth and multiply",
                replacement=self.replace_node(node, new_node),
            )
        self.generic_visit(node)


class TestHouseDivided(BaseNodeVisitorTester):
    visitor_cls = HouseDivided

    def test_once(self):
        self.assert_is_changed("50 / 2\n", "50 * 2\n")
        self.assert_is_changed("50 * 2\n", "50 ** 2\n")

    def test_repeat(self):
        self.assert_is_changed("50 / 2\n", "50 ** 2\n", repeat=True)


def assert_code_equal(expected, actual):
    """Asserts that two pieces of code are equal, and prints a nice diff if they are not."""
    # In Python2.7 ast_decompiler sometimes inserts an extra newline in the beginning
    # for some reason. We don't care.
    expected = expected.lstrip()
    actual = actual.lstrip()
    if expected != actual:
        diff = "".join(
            line + "\n"
            for line in difflib.unified_diff(
                expected.splitlines(), actual.splitlines(), "expected", "actual"
            )
        )
        message = """expected != actual
>>> expected:
%s
>>> actual:
%s
>>> diff:
%s
""" % (
            expected,
            actual,
            diff,
        )
        assert False, message


# Helpers for excluding tests depending on Python version
def _dummy_function(*args, **kwargs):
    return


def only_before(version):
    """Decorator to run a test only before a certain version.

    Example usage:

        @only_before((3, 0))
        def test_xrange():
            xrange(1)

    """

    def decorator(fn):
        if sys.version_info < version:
            return fn
        else:
            return _dummy_function

    return decorator


def skip_before(version):
    """Decorator to skip a test only before a certain version.

    Example usage:

        @skip_before((3, 0))
        def test_huge_range():
            range(1000000000000000000000)

    """

    def decorator(fn):
        if sys.version_info < version:
            return _dummy_function
        else:
            return fn

    return decorator
