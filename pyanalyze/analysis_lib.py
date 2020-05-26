from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

"""

Commonly useful components for static analysis tools.

"""

import ast
import qcore
import os


def _all_files(root, filter_function=None):
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


def files_with_extension_from_directory(extension, dirname):
    """Finds all files in a given directory with this extension."""
    return _all_files(dirname, filter_function=lambda fn: fn.endswith("." + extension))


def get_indentation(line):
    """Returns the indendation of a line of code."""
    if len(line.lstrip()) == 0:
        # if it is a newline or a line with just spaces
        return 0
    return len(line) - len(line.lstrip())


def get_line_range_for_node(node, lines):
    """Returns the lines taken up by a Python ast node.

    lines is a list of code lines for the file the node came from.

    """
    first_lineno = node.lineno
    # iterate through all childnodes and find the max lineno
    last_lineno = first_lineno + 1
    for childnode in ast.walk(node):
        if hasattr(childnode, "end_lineno"):
            last_lineno = max(last_lineno, childnode.end_lineno)
        elif hasattr(childnode, "lineno"):
            last_lineno = max(last_lineno, childnode.lineno)

    def is_part_of_same_node(first_line, line):

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


def safe_text(obj):
    """Safely turns an object into a textual representation.

    Calls str(), then on Python 2 decodes the result.

    """
    result = qcore.safe_str(obj)
    if isinstance(result, bytes):
        try:
            result = result.decode("utf-8")
        except Exception as e:
            result = "<n/a: .decode() raised %r>" % e
    return result


def is_iterable(obj):
    """Returns whether a Python object is iterable."""
    typ = type(obj)
    if hasattr(typ, "__iter__"):
        return True
    return hasattr(typ, "__getitem__") and hasattr(typ, "__len__")
