import ast
from pathlib import Path
from typing import Callable, Type

from .ast_annotator import annotate_file, annotate_code
from .analysis_lib import files_with_extension_from_directory
from .value import KnownValue, Value


def _check_inferred_value(
    tree: ast.Module,
    node_type: Type[ast.AST],
    value: Value,
    predicate: Callable[[ast.AST], bool] = lambda node: True,
) -> None:
    for node in ast.walk(tree):
        if isinstance(node, node_type) and predicate(node):
            assert hasattr(node, "inferred_value"), repr(node)
            assert value == node.inferred_value, ast.dump(node)


def test_annotate_code() -> None:
    tree = annotate_code("a = 1")
    _check_inferred_value(tree, ast.Constant, KnownValue(1))
    _check_inferred_value(tree, ast.Name, KnownValue(1))

    tree = annotate_code(
        """
class X:
    def __init__(self):
        self.a = 1
    """
    )
    _check_inferred_value(tree, ast.Attribute, KnownValue(1))
    tree = annotate_code(
        """
class X:
    def __init__(self):
        self.a = 1

x = X()
x.a + 1
    """
    )
    _check_inferred_value(tree, ast.BinOp, KnownValue(2))

    tree = annotate_code(
        """
class A:
    def __init__(self):
        self.a = 1

    def bla(self):
        return self.a


a = A()
b = a.bla()
"""
    )
    _check_inferred_value(tree, ast.Name, KnownValue(1), lambda node: node.id == "b")


def test_everything_annotated() -> None:
    pyanalyze_dir = Path(__file__).parent
    failures = []
    for filename in sorted(files_with_extension_from_directory("py", pyanalyze_dir)):
        tree = annotate_file(filename, show_errors=True)
        for node in ast.walk(tree):
            if (
                hasattr(node, "lineno")
                and not hasattr(node, "inferred_value")
                and not isinstance(node, (ast.keyword, ast.arg))
            ):
                failures.append((filename, node))
    if failures:
        for filename, node in failures:
            print(f"{filename}:{node.lineno}:{node.col_offset}: {ast.dump(node)}")
        assert False, f"found no annotations on {len(failures)} expressions"
