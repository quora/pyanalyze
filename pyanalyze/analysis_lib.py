"""

Commonly useful components for static analysis tools.

"""
import ast
import linecache
import os
from pathlib import Path
import secrets
import sys
import types
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Set, Union


def _all_files(
    root: Union[str, Path], filter_function: Optional[Callable[[str], bool]] = None
) -> Set[str]:
    """Returns the set of all files at the given root.

    Filtered optionally by the filter_function.

    """
    all_files = set()
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filter_function is not None and not filter_function(filename):
                continue
            all_files.add(os.path.join(dirpath, filename))
    return all_files


def files_with_extension_from_directory(
    extension: str, dirname: Union[str, Path]
) -> Set[str]:
    """Finds all files in a given directory with this extension."""
    return _all_files(dirname, filter_function=lambda fn: fn.endswith("." + extension))


def get_indentation(line: str) -> int:
    """Returns the indendation of a line of code."""
    if len(line.lstrip()) == 0:
        # if it is a newline or a line with just spaces
        return 0
    return len(line) - len(line.lstrip())


def get_line_range_for_node(node: ast.AST, lines: List[str]) -> List[int]:
    """Returns the lines taken up by a Python ast node.

    lines is a list of code lines for the file the node came from.

    """
    first_lineno = node.lineno
    # iterate through all childnodes and find the max lineno
    last_lineno = first_lineno + 1
    for childnode in ast.walk(node):
        end_lineno = getattr(childnode, "end_lineno", None)
        if end_lineno is not None:
            last_lineno = max(last_lineno, end_lineno)
        elif hasattr(childnode, "lineno"):
            last_lineno = max(last_lineno, childnode.lineno)

    def is_part_of_same_node(first_line: str, line: str) -> bool:
        current_indent = get_indentation(line)
        first_indent = get_indentation(first_line)
        if current_indent > first_indent:
            return True
        # because closing parenthesis are at the same indentation
        # as the expression
        line = line.lstrip()
        if len(line) == 0:
            # if it is just a newline then the node has likely ended
            return False
        if current_indent == first_indent and line.lstrip()[0] in [")", "]", "}"]:
            return True
        # probably part of the same multiline string
        for multiline_delim in ('"""', "'''"):
            if multiline_delim in first_line and line.strip() == multiline_delim:
                return True
        return False

    first_line = lines[first_lineno - 1]

    while last_lineno - 1 < len(lines) and is_part_of_same_node(
        first_line, lines[last_lineno - 1]
    ):
        last_lineno += 1
    return list(range(first_lineno, last_lineno))


@dataclass
class _FakeLoader:
    source: str

    def get_source(self, name: object) -> str:
        return self.source


def make_module(
    code_str: str, extra_scope: Mapping[str, object] = {}
) -> types.ModuleType:
    """Creates a Python module with the given code."""
    # Make the name unique to avoid clobbering the overloads dict
    # from pyanalyze.extensions.overload.
    token = secrets.token_hex()
    module_name = f"<test input {secrets.token_hex()}>"
    filename = f"{token}.py"
    mod = types.ModuleType(module_name)
    scope = mod.__dict__
    scope["__name__"] = module_name
    scope["__file__"] = filename
    scope["__loader__"] = _FakeLoader(code_str)

    # This allows linecache later to retrieve source code
    # from this module, which helps the type evaluator.
    linecache.lazycache(filename, scope)
    scope.update(extra_scope)
    code = compile(code_str, filename, "exec")
    exec(code, scope)
    sys.modules[module_name] = mod
    return mod


def is_positional_only_arg_name(name: str, class_name: Optional[str] = None) -> bool:
    # https://www.python.org/dev/peps/pep-0484/#positional-only-arguments
    # Work around Python's name mangling
    if class_name is not None:
        prefix = f"_{class_name}"
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.startswith("__") and not name.endswith("__")


def get_attribute_path(node: ast.AST) -> Optional[List[str]]:
    """Gets the full path of an attribute lookup.

    For example, the code string "a.model.question.Question" will resolve to the path
    ['a', 'model', 'question', 'Question']. This is used for comparing such paths to
    lists of functions that we treat specially.

    """
    if isinstance(node, ast.Name):
        return [node.id]
    elif isinstance(node, ast.Attribute):
        root_value = get_attribute_path(node.value)
        if root_value is None:
            return None
        root_value.append(node.attr)
        return root_value
    else:
        return None
