import ast
from qcore.asserts import assert_eq

from .analysis_lib import get_indentation, get_line_range_for_node


def test_get_indentation() -> None:
    assert_eq(0, get_indentation("\n"))
    assert_eq(0, get_indentation(""))
    assert_eq(4, get_indentation("    pass\n"))
    assert_eq(1, get_indentation(" hello"))


CODE = r'''from qcore.asserts import assert_eq

from pyanalyze.analysis_lib import get_indentation


def test_get_indentation() -> None:
    assert_eq(0, get_indentation('\n'))
    assert_eq(0, get_indentation(''))
    assert_eq(4, get_indentation('    pass\n'))
    assert_eq(1, get_indentation(' hello'))


def test_get_line_range_for_node() -> None:
    pass

x = """
really
long
multiline
string
"""
'''


def test_get_line_range_for_node() -> None:
    lines = CODE.splitlines()
    tree = ast.parse(CODE)
    assert_eq([1], get_line_range_for_node(tree.body[0], lines))
    assert_eq([3], get_line_range_for_node(tree.body[1], lines))
    assert_eq([6, 7, 8, 9, 10], get_line_range_for_node(tree.body[2], lines))
    assert_eq([13, 14], get_line_range_for_node(tree.body[3], lines))
    assert_eq([16, 17, 18, 19, 20, 21], get_line_range_for_node(tree.body[4], lines))
