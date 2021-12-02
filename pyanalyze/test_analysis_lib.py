import ast

from .analysis_lib import get_indentation, get_line_range_for_node


def test_get_indentation() -> None:
    assert 0 == get_indentation("\n")
    assert 0 == get_indentation("")
    assert 4 == get_indentation("    pass\n")
    assert 1 == get_indentation(" hello")


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
    assert [1] == get_line_range_for_node(tree.body[0], lines)
    assert [3] == get_line_range_for_node(tree.body[1], lines)
    assert [6, 7, 8, 9, 10] == get_line_range_for_node(tree.body[2], lines)
    assert [13, 14] == get_line_range_for_node(tree.body[3], lines)
    assert [16, 17, 18, 19, 20, 21] == get_line_range_for_node(tree.body[4], lines)
