"""

Base class for scripts that run ast-based checks on the codebase.

"""

import argparse
from ast_decompiler import decompile
import ast
import codemod
import collections
from contextlib import contextmanager
import concurrent.futures
from dataclasses import dataclass
from enum import Enum
import qcore
import cProfile
import logging
import os
import os.path
import re
import subprocess
import sys
import tempfile
import builtins
from builtins import print as real_print
from types import ModuleType
from typing_extensions import TypedDict
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from . import analysis_lib


@dataclass(frozen=True)
class _FakeNode:
    lineno: int
    col_offset: int


# Environment variable that may contain the name of a file listing Python files that we should run
# tests on. This is used in test-local for running tests like test_scope only on modified files.
FILE_ENVIRON_KEY = "ANS_STATIC_ANALYSIS_FILE"

# If this comment occurs in a line with an error, or if the line before the error contains exactly
# this comment, the error is ignored.
IGNORE_COMMENT = "# static analysis: ignore"

# Upper limit on the number of iterations when repeat_until_no_errors is enabled
# to guard against infinite loop
ITERATION_LIMIT = 150

UNUSED_OBJECT_FILENAME = "<unused>"


class _PatchWithDescription(codemod.Patch):
    def __init__(
        self,
        start_line_number: int,
        end_line_number: Optional[int] = None,
        new_lines: Optional[List[str]] = None,
        path: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        super().__init__(start_line_number, end_line_number, new_lines, path)
        self.description = description

    def render_range(self) -> str:
        text = super(_PatchWithDescription, self).render_range()
        if self.description is not None:
            return text + ": " + self.description
        return text


@dataclass
class _Query:
    """Simple equivalent of codemod.Query."""

    patches: List[_PatchWithDescription]

    def generate_patches(self) -> List[_PatchWithDescription]:
        return self.patches


class VisitorError(Exception):
    def __init__(self, message: str, error_code: Optional[Enum] = None) -> None:
        self.message = message
        self.error_code = error_code

    def __str__(self) -> str:
        return self.message


@dataclass
class Replacement:
    """Simple class that contains replacement info for the --autofix option.

    Also contains the error string for the case when test_scope just shows the
    error on stdout (i.e. no autofix)

    linenos_to_delete: line numbers to delete

    lines_to_add: list of strings (lines) to add. These are added right after the
    last deleted line.

    error_str: error to show on stdout

    """

    linenos_to_delete: Sequence[int]
    lines_to_add: Optional[Sequence[str]]
    error_str: Optional[str] = None


class FileNotFoundError(Exception):
    pass


class Failure(TypedDict, total=False):
    description: str
    filename: str
    code: Enum
    lineno: int
    context: str
    message: str


class BaseNodeVisitor(ast.NodeVisitor):
    """Base Visitor class that can run on all files in a/ and show detailed error messages."""

    # Number of context lines to show around errors
    CONTEXT_LINES: int = 3
    error_code_enum: Optional[Type[Enum]] = None
    default_module: Optional[ModuleType] = None  # module to run on by default
    # whether to look at FILE_ENVIRON_KEY to find files to run on
    should_check_environ_for_files: bool = True

    _changes_for_fixer = collections.defaultdict(list)

    tree: ast.Module
    all_failures: List[Failure]

    def __init__(
        self,
        filename: str,
        contents: str,
        tree: ast.Module,
        settings: Optional[Mapping[Enum, bool]] = None,
        fail_after_first: bool = False,
        verbosity: int = logging.CRITICAL,
        collect_failures: bool = False,
        add_ignores: bool = False,
    ) -> None:
        """Constructor.

        filename: name of the file to run on (to show in error messages)
        contents: code that the visitor is run on
        fail_after_first: whether to throw an error after the first problem is detected
        verbosity: controls how much logging is emitted

        """
        if not isinstance(contents, str):
            raise TypeError("File contents must be text, not {}".format(type(contents)))
        super(BaseNodeVisitor, self).__init__()
        self.filename = filename
        self.contents = contents
        self.tree = tree
        self.fail_after_first = fail_after_first
        self._logging_level = verbosity
        self.logger = logging.Logger(self.__class__.__name__, level=verbosity)
        self.logger.addHandler(logging.StreamHandler())
        self.settings = settings
        self.all_failures = []
        self.used_ignores = set()
        self.seen_errors = set()  # of (node, error_code) pairs
        self.add_ignores = add_ignores
        self.caught_errors = None

    def check(self) -> List[Failure]:
        """Runs the class's checks on a tree."""
        self.log(logging.INFO, "Check file", self.filename)
        self.visit(self.tree)
        return self.all_failures

    def check_for_test(self, apply_changes: bool = False) -> Any:
        """Entry point for testing. Also applies all changes if apply_changes is on."""
        if not apply_changes:
            return self.check()
        changes = collections.defaultdict(list)
        with qcore.override(self.__class__, "_changes_for_fixer", changes):
            result = self.check()
        lines = [line + "\n" for line in self.contents.splitlines()]
        if self.filename in changes:
            lines = self._apply_changes_to_lines(changes[self.filename], lines)
        return result, "".join(lines)

    def log(self, level: int, label: str, value: object) -> None:
        if level < self._logging_level:
            return
        self.logger.log(
            level, "%s: %s" % (qcore.safe_str(label), qcore.safe_str(value))
        )

    @qcore.caching.cached_per_instance()
    def _lines(self) -> List[str]:
        return [line + "\n" for line in self.contents.splitlines()]

    @qcore.caching.cached_per_instance()
    def has_file_level_ignore(
        self, error_code: Optional[Enum] = None, ignore_comment: str = IGNORE_COMMENT
    ) -> bool:
        # if the IGNORE_COMMENT occurs before any non-comment line, all errors in the file are
        # ignored
        for i, line in enumerate(self._lines()):
            if not line.startswith("#"):
                return False
            if (
                line.strip() == ignore_comment
                or error_code is not None
                and line.strip() == f"{ignore_comment}[{error_code.name}]"
            ):
                self.used_ignores.add(i)
                return True
        return False

    def get_unused_ignores(self) -> List[Tuple[int, str]]:
        """Returns line numbers and lines that have unused ignore comments."""
        return [
            (i, line)
            for i, line in enumerate(self._lines())
            if IGNORE_COMMENT in line and i not in self.used_ignores
        ]

    def show_errors_for_unused_ignores(self, error_code: Enum) -> None:
        """Shows errors for any unused ignore comments."""
        for i, line in self.get_unused_ignores():
            node = _FakeNode(i + 1, line.index(IGNORE_COMMENT))
            stripped = line.strip()
            if stripped == IGNORE_COMMENT or re.match(
                r"^%s\[[^\s\]]+\]$" % (re.escape(IGNORE_COMMENT),), stripped
            ):
                # just remove the line
                replacement = Replacement([i + 1], [])
            else:
                rgx = re.compile(r"%s(\[[^\s\]]+\])?" % (re.escape(IGNORE_COMMENT),))
                replacement = Replacement([i + 1], [rgx.sub("", line)])
            self.show_error(
                node, error_code=error_code, replacement=replacement, obey_ignore=False
            )

    def show_errors_for_bare_ignores(self, error_code: Enum) -> None:
        """Shows errors for ignore comments without an error code."""
        if self.has_file_level_ignore():
            # file-level ignores are allowed to be blanket ignores
            return
        for i, line in enumerate(self._lines()):
            if IGNORE_COMMENT in line and IGNORE_COMMENT + "[" not in line:
                node = _FakeNode(i + 1, line.index(IGNORE_COMMENT))
                self.show_error(node, error_code=error_code, obey_ignore=False)

    @classmethod
    def check_file(
        cls,
        filename: str,
        assert_passes: bool = True,
        include_tests: bool = False,
        **kwargs: Any,
    ) -> List[Failure]:
        """Run checks on a single file.

        include_tests and assert_passes are arguments here for compatibility with check_all_files.

        """
        try:
            with open(filename, "r", encoding="utf-8") as f:
                contents = f.read()
        except IOError:
            raise FileNotFoundError(repr(filename))
        except UnicodeDecodeError:
            raise FileNotFoundError("Failed to decode contents of {}".format(filename))
        tree = ast.parse(contents.encode("utf-8"), filename)
        return cls(filename, contents, tree, **kwargs).check()

    @classmethod
    def check_all_files(
        cls, include_tests: bool = False, assert_passes: bool = True, **kwargs: Any
    ) -> List[Failure]:
        """Runs the check for all files in scope or changed files if we are test-local."""
        if "settings" not in kwargs:
            kwargs["settings"] = cls._get_default_settings()
        kwargs = cls.prepare_constructor_kwargs(kwargs)
        files = cls.get_files_to_check(include_tests)
        all_failures = cls._run_on_files(files, **kwargs)
        if assert_passes:
            assert not all_failures, "".join(
                failure["message"] for failure in all_failures
            )
        return all_failures

    @classmethod
    def get_files_to_check(cls, include_tests: bool) -> List[str]:
        """Produce the list of files to check."""
        if cls.should_check_environ_for_files:
            environ_files = get_files_to_check_from_environ()
        else:
            environ_files = None
        if environ_files is not None:
            return [
                filename
                for filename in environ_files
                if not cls._should_ignore_module(filename)
                and not filename.endswith(".so")
            ]
        else:
            return sorted(set(cls._get_all_python_files(include_tests=include_tests)))

    @classmethod
    def prepare_constructor_kwargs(cls, kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
        return kwargs

    @classmethod
    def main(cls) -> int:
        """Can be used as a main function. Calls the checker on files given on the command line."""
        args = cls._get_argument_parser().parse_args()

        if cls.error_code_enum is not None:
            if args.enable_all:
                settings = {code: True for code in cls.error_code_enum}
            elif args.disable_all:
                settings = {code: False for code in cls.error_code_enum}
            else:
                settings = cls._get_default_settings()
            if settings is not None:
                for setting in args.enable:
                    settings[cls.error_code_enum[setting]] = True
                for setting in args.disable:
                    settings[cls.error_code_enum[setting]] = False
            kwargs = {
                key: value
                for key, value in args.__dict__.items()
                if key not in {"enable_all", "disable_all", "enable", "disable"}
            }
            kwargs["settings"] = settings
        else:
            kwargs = dict(args.__dict__)
        markdown_output = kwargs.pop("markdown_output", None)

        verbose = kwargs.pop("verbose", 0)
        if verbose == 0 or verbose is None:
            verbosity = logging.ERROR
        elif verbose == 1:
            verbosity = logging.INFO
        else:
            verbosity = logging.DEBUG
        kwargs["verbosity"] = verbosity

        # when run as main() don't throw an error if something failed
        kwargs["assert_passes"] = False

        run_fixer = kwargs.pop("run_fixer", False)
        autofix = kwargs.pop("autofix", False)
        repeat_until_no_errors = kwargs.pop("repeat_until_no_errors", False)
        num_iterations = kwargs.pop("num_iterations", 1)
        kwargs = cls.prepare_constructor_kwargs(kwargs)
        if repeat_until_no_errors:
            iteration = 0
            print("Running iteration 0")
            while cls._run_and_apply_changes(kwargs, autofix=True):
                iteration += 1
                # if num_iterations is 1, then it's just the default value
                if num_iterations != 1 and iteration >= num_iterations:
                    break
                assert iteration <= ITERATION_LIMIT, "Iteration Limit Exceeded!"
                print(f"Running iteration {iteration}")
            failures = []
        elif run_fixer or autofix:
            failures = cls._run_and_apply_changes(kwargs, autofix=autofix)
        else:
            failures = cls._run(**kwargs)
            if markdown_output is not None and failures:
                cls._write_markdown_report(markdown_output, failures)
        return 1 if failures else 0

    @classmethod
    def _write_markdown_report(cls, output_file: str, failures: List[Failure]) -> None:
        by_file = collections.defaultdict(list)
        for failure in failures:
            by_file[failure["filename"]].append(failure)

        prefix = os.path.commonpath(
            [key for key in by_file if key != UNUSED_OBJECT_FILENAME]
        )

        with open(output_file, "w") as f:
            f.write("%d total failures in %d files\n\n" % (len(failures), len(by_file)))

            for filename, file_failures in sorted(by_file.items()):
                if filename != UNUSED_OBJECT_FILENAME:
                    filename = filename[len(prefix) :]
                f.write(f"\n### {filename} ({len(file_failures)} failures)\n\n")

                for failure in sorted(
                    file_failures, key=lambda failure: failure.get("lineno", 0)
                ):
                    if "\n" in failure["message"].strip():
                        lines = failure["message"].splitlines()[1:]
                        f.write("* line %s: `%s`\n" % (failure.get("lineno"), lines[0]))
                        f.write("```\n")
                        for line in lines[1:]:
                            f.write(f"{line}\n")
                        f.write("```\n")
                    else:
                        f.write("* `%s`" % (failure["message"].strip(),))

    @classmethod
    def _run_and_apply_changes(
        cls, kwargs: Mapping[str, Any], autofix: bool = False
    ) -> bool:
        changes = collections.defaultdict(list)
        with qcore.override(cls, "_changes_for_fixer", changes):
            try:
                had_failure = bool(cls._run(**kwargs))
            except VisitorError:
                had_failure = True
        # ignore run_fixer if autofix is enabled
        if autofix:
            cls._apply_changes(changes)
        else:
            patches = []
            for filename in changes:
                for change in changes[filename]:
                    linenos = sorted(change.linenos_to_delete)
                    additions = change.lines_to_add
                    if len(linenos) > 1:
                        start_lineno, end_lineno = linenos[0], linenos[-1]
                    else:
                        start_lineno, end_lineno = linenos[0], linenos[0]
                    patches.append(
                        _PatchWithDescription(
                            start_lineno - 1,
                            end_lineno,
                            new_lines=additions,
                            path=filename,
                            description=change.error_str,
                        )
                    )
            if patches:
                # poor man's version of https://github.com/facebook/codemod/pull/113
                with qcore.override(builtins, "print", _flushing_print):
                    codemod.run_interactive(_Query(patches))
        return had_failure

    @classmethod
    def _apply_changes(cls, changes: Dict[str, List[Replacement]]) -> None:
        for filename, changeset in changes.items():
            with open(filename, "r") as f:
                lines = f.readlines()
            lines = cls._apply_changes_to_lines(changeset, lines)
            with open(filename, "w") as f:
                f.write("".join(lines))

    @classmethod
    def _apply_changes_to_lines(
        cls, changes: List[Replacement], input_lines: Sequence[str]
    ) -> Sequence[str]:
        # only apply the first change because that change might affect other fixes
        # that test_scope came up for that file. So we break after finding first applicable fix.
        lines = list(input_lines)
        if changes:
            change = changes[0]
            additions = change.lines_to_add
            if additions is not None:
                lines_to_remove = change.linenos_to_delete
                max_line = max(lines_to_remove)
                # add the additions after the max_line
                lines = [*lines[:max_line], *additions, *lines[max_line:]]
                lines_to_remove = sorted(lines_to_remove, reverse=True)
                for lineno in lines_to_remove:
                    del lines[lineno - 1]
        return lines

    @classmethod
    def _get_default_settings(cls) -> Optional[Dict[Enum, bool]]:
        if cls.error_code_enum is None:
            return None
        else:
            return {
                code: cls.is_enabled_by_default(code) for code in cls.error_code_enum
            }

    @contextmanager
    def catch_errors(self) -> Iterator[List[Dict[str, Any]]]:
        caught_errors = []
        with qcore.override(self, "caught_errors", caught_errors):
            yield caught_errors

    def show_caught_errors(self, errors: Iterable[Dict[str, Any]]) -> None:
        for error in errors:
            self.show_error(**error)

    def show_error(
        self,
        node: Union[ast.AST, _FakeNode, None],
        e: Optional[str] = None,
        error_code: Optional[Enum] = None,
        *,
        replacement: Optional[Replacement] = None,
        obey_ignore: bool = True,
        ignore_comment: str = IGNORE_COMMENT,
        detail: Optional[str] = None,
    ) -> Optional[Failure]:
        """Shows an error associated with this node.

        Arguments:
        - node: AST node to show the error on. May be None if there is no
          associated AST node.
        - e: error message. If not given, it defaults to the description of the error_code.
        - error_code: error code for this error; it is used to not show the error when the user
          has disabled it.
        - replacement: Object of class Replacement used as a suggested replacement
          of part of the code.
        - obey_ignore: if True, we obey ignore_comment on individual lines. (We always obey
          file-level ignore comments.)
        - ignore_comment: Comment that can be used to ignore this error. (By default, this
          is "# static analysis: ignore".)

        """
        if self.caught_errors is not None:
            self.caught_errors.append(
                {
                    "node": node,
                    "e": e,
                    "error_code": error_code,
                    "replacement": replacement,
                    "obey_ignore": obey_ignore,
                    "ignore_comment": ignore_comment,
                    "detail": detail,
                }
            )
            return None

        # check if error was disabled
        if self.settings is not None and error_code is not None:
            if not self.settings.get(error_code, True):
                return None

        if self.has_file_level_ignore(error_code, ignore_comment):
            return None

        key = (node, error_code or e)
        if key in self.seen_errors:
            self.logger.info("Ignoring duplicate error %s", key)
            return None
        self.seen_errors.add(key)

        if e is None:
            assert (
                error_code is not None
            ), "Must specify an error message or an error code"
            e = self.get_description_for_error_code(error_code)

        if node:
            lineno = node.lineno
            col_offset = node.col_offset
        else:
            lineno = col_offset = None

        # https://github.com/quora/pyanalyze/issues/112
        error = cast(Failure, {"description": str(e), "filename": self.filename})
        message = f"\n{e}"
        if error_code is not None:
            error["code"] = error_code
            message += f" (code: {error_code.name})"
        if detail is not None:
            message += f"\n{detail}"
        if lineno is not None:
            error["lineno"] = lineno
            message += f"\nIn {self.filename} at line {lineno}\n"
        else:
            message += f"\n In {self.filename}"
        lines = self._lines()

        if obey_ignore and lineno is not None:
            this_line = lines[lineno - 1]
            if (
                re.search("%s(?!\\[)" % (re.escape(ignore_comment),), this_line)
                or error_code is not None
                and f"{ignore_comment}[{error_code.name}]" in this_line
            ):
                self.used_ignores.add(lineno - 1)
                return
            prev_line = lines[lineno - 2].strip()
            if (
                prev_line == ignore_comment
                or error_code is not None
                and prev_line == f"{ignore_comment}[{error_code.name}]"
            ):
                self.used_ignores.add(lineno - 2)
                return

        self.had_failure = True

        if lineno is not None:
            context = ""
            min_line = max(lineno - self.CONTEXT_LINES, 1)
            max_line = min(lineno + self.CONTEXT_LINES + 1, len(lines) + 1)
            for i in range(min_line, max_line):
                # four columns for the line number
                # app/view/question/__init__.py is 6900 lines
                context += "%4d: %s" % (i, lines[i - 1])
                if i == lineno and col_offset is not None:
                    # caret to indicate the position of the error
                    context += " " * (6 + col_offset) + "^\n"
            message += context
            error["context"] = context

        if lineno is not None and self._changes_for_fixer is not None:
            if self.add_ignores:
                this_line = lines[lineno - 1]
                indentation = analysis_lib.get_indentation(this_line)
                if error_code is not None:
                    ignore = f"{ignore_comment}[{error_code.name}]"
                else:
                    ignore = ignore_comment
                replacement = Replacement(
                    [lineno],
                    ["%s%s\n" % (" " * indentation, ignore), this_line],
                    str(e),
                )
            else:
                if replacement is not None:
                    replacement.error_str = str(e)
                else:
                    replacement = Replacement([lineno], None, str(e))
            self._changes_for_fixer[self.filename].append(replacement)

        error["message"] = message
        self.all_failures.append(error)
        sys.stderr.write(message)
        sys.stderr.flush()
        if self.fail_after_first:
            raise VisitorError(message, error_code)
        return error

    def _get_attribute_path(self, node: ast.AST) -> Optional[List[str]]:
        """Gets the full path of an attribute lookup.

        For example, the code string "a.model.question.Question" will resolve to the path
        ['a', 'model', 'question', 'Question']. This is used for comparing such paths to
        lists of functions that we treat specially.

        """
        if isinstance(node, ast.Name):
            return [node.id]
        elif isinstance(node, ast.Attribute):
            root_value = self._get_attribute_path(node.value)
            if root_value is None:
                return None
            root_value.append(node.attr)
            return root_value
        else:
            return None

    @classmethod
    def _run(
        cls, profile: bool = False, num_iterations: int = 1, **kwargs: Any
    ) -> Optional[List[Failure]]:
        result = None
        for _ in range(num_iterations):
            if profile:
                with _Profile():
                    result = cls._run_on_files_or_all(**kwargs)
            else:
                result = cls._run_on_files_or_all(**kwargs)
        return result

    @classmethod
    def _run_on_files_or_all(
        cls, files: Optional[Sequence[str]], **kwargs: Any
    ) -> List[Failure]:
        files = files or cls.get_default_directories()
        if not files:
            return cls.check_all_files(**kwargs)
        else:
            return cls._run_on_files(_get_all_files(files), **kwargs)

    @classmethod
    def _run_on_files(cls, files: Iterable[str], **kwargs: Any) -> List[Failure]:
        all_failures = []
        args = [(filename, kwargs) for filename in sorted(files)]
        if kwargs.pop("parallel", False):
            extra_data = []
            with concurrent.futures.ProcessPoolExecutor(os.cpu_count()) as executor:
                for failures, extra in executor.map(cls._check_file_single_arg, args):
                    all_failures += failures
                    extra_data.append(extra)
            cls.merge_extra_data(extra_data, **kwargs)
        else:
            for failures, _ in map(cls._check_file_single_arg, args):
                all_failures += failures
        return all_failures

    @classmethod
    def _check_file_single_arg(
        cls, args: Tuple[str, Dict[str, Any]]
    ) -> Tuple[List[Failure], Any]:
        filename, kwargs = args
        main_module = sys.modules["__main__"]
        try:
            return cls.check_file_in_worker(filename, **kwargs)
        finally:
            # Some modules cause __main__ to get reassigned for unclear reasons. So let's put it back.
            sys.modules["__main__"] = main_module

    @classmethod
    def check_file_in_worker(
        cls, filename: str, **kwargs: Any
    ) -> Tuple[List[Failure], Any]:
        """Checks a single file in a parallel worker.

        Returns a tuple of (failures, extra data). The extra data will be passed to
        merge_extra_data() once we finish processing all files.

        By default the extra data is None. Override this in a subclass to aggregate
        data from the parallel workers at the end of the run.

        """
        failures = cls.check_file(filename, **kwargs)
        return failures, None

    @classmethod
    def merge_extra_data(cls, extra_data: object, **kwargs: Any) -> None:
        """Override this to aggregate data passed from parallel workers."""
        pass

    @classmethod
    def _get_argument_parser(cls) -> argparse.ArgumentParser:
        """Returns an argument parser for this visitor.

        Override this to add additional arguments. Arguments are passed as kwargs
        to _run and eventually to the class constructor.

        """
        if cls.error_code_enum is not None:
            code_descriptions = []
            for code in cls.error_code_enum:
                enabled_string = "on" if cls.is_enabled_by_default(code) else "off"
                code_descriptions.append(
                    "  - %s: %s (default: %s)"
                    % (
                        code.name,
                        cls.get_description_for_error_code(code),
                        enabled_string,
                    )
                )

            epilog = "Supported checks:\n" + "\n".join(code_descriptions)
        else:
            epilog = None

        parser = argparse.ArgumentParser(
            description=cls.__doc__,
            epilog=epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "files", nargs="*", default=".", help="Files or directories to check"
        )
        parser.add_argument(
            "-v", "--verbose", help="Print more information.", action="count"
        )
        parser.add_argument(
            "-f",
            "--run-fixer",
            help="Interactively fix all errors found.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--profile",
            help="Run with profiling enabled",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "-n",
            "--num-iterations",
            help="Number of iterations to run",
            type=int,
            default=1,
        )
        parser.add_argument(
            "-A",
            "--autofix",
            help="Automatically do the recommended fixes.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "-r",
            "--repeat-until-no-errors",
            help=(
                "Repeatedly runs in autofix mode until no errors are encountered. Use"
                " this wisely."
            ),
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--fail-after-first",
            help="Stop at the first failure.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--parallel",
            help="Check files in parallel",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--markdown-output",
            help=(
                "Write errors to this file in markdown format. "
                "Suitable for summarizing and tracking errors."
            ),
        )
        parser.add_argument(
            "--add-ignores",
            help=(
                "Add ignore comments for all errors detected. "
                "Must be used with -f/--run-fixer."
            ),
            action="store_true",
            default=False,
        )

        if cls.error_code_enum is not None:
            all_group = parser.add_mutually_exclusive_group()
            all_group.add_argument(
                "-a",
                "--enable-all",
                action="store_true",
                default=False,
                help="Enable all checks by default",
            )
            all_group.add_argument(
                "--disable-all",
                action="store_true",
                default=False,
                help="Disable all checks by default",
            )
            parser.add_argument(
                "-e",
                "--enable",
                help="Enable a check",
                action="append",
                default=[],
                choices=[code.name for code in cls.error_code_enum],
            )
            parser.add_argument(
                "-d",
                "--disable",
                help="Disable a check",
                action="append",
                default=[],
                choices=[code.name for code in cls.error_code_enum],
            )
        return parser

    @classmethod
    def is_enabled_by_default(cls, code: Enum) -> bool:
        return True

    @classmethod
    def get_description_for_error_code(cls, code: Enum) -> str:
        return f"Error: {code}"

    @classmethod
    def get_default_modules(cls) -> Sequence[ModuleType]:
        if cls.default_module is not None:
            return (cls.default_module,)
        return ()

    @classmethod
    def get_default_directories(cls) -> Sequence[str]:
        return ()

    @classmethod
    def _get_all_python_files(
        cls, include_tests: bool = False, modules: Optional[Iterable[ModuleType]] = None
    ) -> Iterable[str]:
        """Gets Python files inside of the given modules that should be tested.

        This includes modules imported as submodules of the module and (if include_tests) test
        modules. We cannot just use all Python files because some standalone scripts cannot be
        imported.

        By default, gives all Python files in the modules returned by get_default_modules.

        """
        if modules is None:
            dirs = cls.get_default_directories()
            if dirs:
                for filename in _get_all_files(dirs):
                    yield filename
                return
            modules = cls.get_default_modules()
        enclosing_module_names = {module.__name__ for module in modules}

        # Iterate over a copy of sys.modules in case we import something
        # while this generator is running.
        for module_name, module in list(sys.modules.items()):
            if module is None:
                continue
            # ignore compiled modules
            if not getattr(module, "__file__", None) or module.__file__.endswith(".so"):
                continue
            if cls._should_ignore_module(module_name):
                continue
            if module_name in enclosing_module_names or any(
                module_name.startswith(f"{enclosing_module_name}.")
                for enclosing_module_name in enclosing_module_names
            ):
                yield module.__file__.rstrip("c")

        if include_tests:
            for module in modules:
                dirname = os.path.dirname(module.__file__)
                cmd = ["find", dirname, "-name", "test*.py"]
                for filename in subprocess.check_output(cmd).split():
                    yield filename

    @classmethod
    def _should_ignore_module(cls, module_name: str) -> bool:
        """Override this to ignore some modules."""
        return False


class NodeTransformer(ast.NodeVisitor):
    """Similar to the standard library's ast.NodeTransformer, but does not mutate in place.

    This makes it possible to reuse AST nodes after they have been passed through this
    Transformer.

    """

    def generic_visit(self, node: ast.AST) -> ast.AST:
        attributes = {}
        for field, _ in ast.iter_fields(node):
            old_value = getattr(node, field, None)
            if isinstance(old_value, list):
                new_value = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_value.extend(value)
                            continue
                    new_value.append(value)
            elif isinstance(old_value, ast.AST):
                new_value = self.visit(old_value)
            else:
                new_value = old_value
            attributes[field] = new_value
        new_node = type(node)(**attributes)
        return ast.copy_location(new_node, node)


class ReplaceNodeTransformer(NodeTransformer):
    def __init__(self, node_to_replace: ast.AST, replacement: ast.AST) -> None:
        self.node_to_replace = node_to_replace
        self.replacement = replacement
        super().__init__()

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if node == self.node_to_replace:
            return self.replacement
        return super().generic_visit(node)


class ReplacingNodeVisitor(BaseNodeVisitor):
    """A NodeVisitor that enables replacing AST nodes directly in errors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(ReplacingNodeVisitor, self).__init__(*args, **kwargs)
        self.current_statement = None

    def replace_node(
        self,
        current_node: ast.AST,
        new_node: ast.AST,
        current_statement: Optional[ast.stmt] = None,
    ) -> Optional[Replacement]:
        if current_statement is None:
            current_statement = self.current_statement
        if current_statement is None:
            return None
        transformer = ReplaceNodeTransformer(current_node, new_node)
        lines = self._lines()
        lines_to_remove = analysis_lib.get_line_range_for_node(current_statement, lines)
        indent = analysis_lib.get_indentation(lines[current_statement.lineno - 1])
        node = transformer.visit(current_statement)
        try:
            parent_lines = decompile(node, starting_indentation=indent).splitlines()
        except NotImplementedError:
            return None
        lines_to_add = [line + "\n" for line in parent_lines]
        return Replacement(lines_to_remove, lines_to_add)

    def remove_node(
        self, current_node: ast.AST, current_statement: Optional[ast.stmt] = None
    ) -> Optional[Replacement]:
        if current_statement is None:
            current_statement = self.current_statement
        if current_statement is None:
            return None
        lines = self._lines()
        lines_to_remove = analysis_lib.get_line_range_for_node(current_statement, lines)
        return Replacement(lines_to_remove, [])

    def visit(self, node: ast.AST) -> Any:
        """Save the node if it is a statement."""
        # This code is also inlined in NameCheckVisitor (a subclass of this class) for speed.
        if isinstance(node, ast.stmt):
            with qcore.override(self, "current_statement", node):
                return super().visit(node)
        else:
            return super().visit(node)


def get_files_to_check_from_environ(
    environ_key: str = FILE_ENVIRON_KEY,
) -> Optional[List[str]]:
    """Returns any files to run on specified in the FILE_ENVIRON_KEY that we should run on.

    If the key isn't in the environ, return None.

    """
    if environ_key in os.environ:
        with open(os.environ[environ_key], "r") as f:
            return [filename.strip() for filename in f]
    else:
        return None


def _get_all_files(lst: Iterable[str]) -> Iterable[str]:
    """Finds all Python files from a list of command-line arguments."""
    for entry in lst:
        if os.path.isdir(entry):
            for filename in sorted(
                analysis_lib.files_with_extension_from_directory("py", entry)
            ):
                yield filename
        else:
            yield entry


def _flushing_print(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("flush", True)
    real_print(*args, **kwargs)


class _Profile(object):
    """Context for profiling an arbitrary block of code."""

    def __init__(self) -> None:
        self.prof = cProfile.Profile()

    def __enter__(self) -> None:
        self.prof.enable()

    def __exit__(self, typ: object, value: object, traceback: object) -> None:
        self.prof.disable()
        self.filename = tempfile.mktemp()
        self.prof.dump_stats(self.filename)
        print("profiler output saved as {}".format(self.filename))
