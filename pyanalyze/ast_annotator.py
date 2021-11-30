"""

Functionality for annotating the AST of a module.

The APIs in this module use pyanalyze's type inference to annotate
an AST with inferred :class:`pyanalyze.value.Value` objects in `.inferred_value`
attributes.

"""
import ast
import os
import traceback
import types
from typing import Optional, Type, Union

from .analysis_lib import make_module
from .error_code import ErrorCode
from .importer import load_module_from_file
from .name_check_visitor import NameCheckVisitor, ClassAttributeChecker
from .find_unused import used


@used  # exposed as an API
def annotate_code(
    code: str,
    *,
    visitor_cls: Type[NameCheckVisitor] = NameCheckVisitor,
    dump: bool = False,
    show_errors: bool = False,
    verbose: bool = False,
) -> ast.Module:
    """Annotate a piece of Python code. Return an AST with extra `inferred_value` attributes.

    Example usage::

        tree = annotate_code("a = 1")
        print(tree.body[0].targets[0].inferred_value)  # Literal[1]

    This will import and ``exec()`` the provided code. If this fails, the code will
    still be annotated but the quality of the annotations will be much lower.

    :param visitor_cls: Pass a subclass of :class:`pyanalyze.name_check_visitor.NameCheckVisitor`
                        to customize pyanalyze behavior.
    :type visitor_cls: Type[NameCheckVisitor]

    :param dump: If True, the annotated AST is printed out.
    :type dump: bool

    :param show_errors: If True, errors from pyanalyze are printed.
    :type show_errors: bool

    :param verbose: If True, more details are printed.
    :type verbose: bool

    """
    tree = ast.parse(code)
    try:
        mod = make_module(code)
    except Exception:
        if verbose:
            traceback.print_exc()
        mod = None
    _annotate_module("", mod, tree, code, visitor_cls, show_errors=show_errors)
    if dump:
        dump_annotated_code(tree)
    return tree


@used  # exposed as an API
def annotate_file(
    path: Union[str, "os.PathLike[str]"],
    *,
    visitor_cls: Type[NameCheckVisitor] = NameCheckVisitor,
    verbose: bool = False,
    dump: bool = False,
    show_errors: bool = False,
) -> ast.AST:
    """Annotate the code in a Python source file. Return an AST with extra `inferred_value` attributes.

    Example usage::

        tree = annotate_file("/some/file.py")
        print(tree.body[0].targets[0].inferred_value)  # Literal[1]

    This will import and exec() the provided code. If this fails, the code will
    still be annotated but the quality of the annotations will be much lower.

    :param visitor_cls: Pass a subclass of :class:`pyanalyze.name_check_visitor.NameCheckVisitor`
                        to customize pyanalyze behavior.
    :type visitor_cls: Type[NameCheckVisitor]

    :param dump: If True, the annotated AST is printed out.
    :type dump: bool

    :param show_errors: If True, errors from pyanalyze are printed.
    :type show_errors: bool

    :param verbose: If True, more details are printed.
    :type verbose: bool

    """
    filename = os.fspath(path)
    try:
        mod, _ = load_module_from_file(filename, verbose=verbose)
    except Exception:
        if verbose:
            traceback.print_exc()
        mod = None

    with open(filename) as f:
        code = f.read()
    tree = ast.parse(code)
    _annotate_module(filename, mod, tree, code, visitor_cls, show_errors=show_errors)
    if dump:
        dump_annotated_code(tree)
    return tree


def dump_annotated_code(
    node: ast.AST, depth: int = 0, field_name: Optional[str] = None
) -> None:
    """Print an annotated AST in a readable format."""
    line = type(node).__name__
    if field_name is not None:
        line = f"{field_name}: {line}"
    if (
        hasattr(node, "lineno")
        and hasattr(node, "col_offset")
        and node.lineno is not None
        and node.col_offset is not None
    ):
        line = f"{line}(@{node.lineno}:{node.col_offset})"
    print(" " * depth + line)
    new_depth = depth + 2
    if hasattr(node, "inferred_value"):
        print(" " * new_depth + str(node.inferred_value))
    for field_name, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            dump_annotated_code(value, new_depth, field_name)
        elif isinstance(value, list):
            if not value:
                continue
            print(" " * new_depth + field_name)
            for element in value:
                if isinstance(element, ast.AST):
                    dump_annotated_code(element, new_depth + 2)
                else:
                    print(" " * (new_depth + 2) + repr(element))
        elif value is not None:
            print(" " * new_depth + f"{field_name}: {value!r}")


def _annotate_module(
    filename: str,
    module: Optional[types.ModuleType],
    tree: ast.Module,
    code_str: str,
    visitor_cls: Type[NameCheckVisitor],
    show_errors: bool = False,
) -> None:
    """Annotate the AST for a module with inferred values.

    Takes the module objects, its AST tree, and its literal code. Modifies the AST object in place.

    """
    with ClassAttributeChecker(visitor_cls.config, enabled=True) as attribute_checker:
        kwargs = visitor_cls.prepare_constructor_kwargs({})
        visitor = visitor_cls(
            filename,
            code_str,
            tree,
            module=module,
            settings={error_code: show_errors for error_code in ErrorCode},
            attribute_checker=attribute_checker,
            annotate=True,
            **kwargs,
        )
        visitor.check(ignore_missing_module=True)
