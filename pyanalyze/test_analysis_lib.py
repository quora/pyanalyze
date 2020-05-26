from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import ast
from qcore.asserts import assert_eq

from pyanalyze.analysis_lib import get_indentation, get_line_range_for_node, is_iterable


def test_get_indentation():
    assert_eq(0, get_indentation("\n"))
    assert_eq(0, get_indentation(""))
    assert_eq(4, get_indentation("    pass\n"))
    assert_eq(1, get_indentation(" hello"))


CODE = r'''from qcore.asserts import assert_eq

from pyanalyze.analysis_lib import get_indentation


def test_get_indentation():
    assert_eq(0, get_indentation('\n'))
    assert_eq(0, get_indentation(''))
    assert_eq(4, get_indentation('    pass\n'))
    assert_eq(1, get_indentation(' hello'))


def test_get_line_range_for_node():
    pass

x = """
really
long
multiline
string
"""
'''


def test_get_line_range_for_node():
    lines = CODE.splitlines()
    tree = ast.parse(CODE)
    assert_eq([1], get_line_range_for_node(tree.body[0], lines))
    assert_eq([3], get_line_range_for_node(tree.body[1], lines))
    assert_eq([6, 7, 8, 9, 10], get_line_range_for_node(tree.body[2], lines))
    assert_eq([13, 14], get_line_range_for_node(tree.body[3], lines))
    assert_eq([16, 17, 18, 19, 20, 21], get_line_range_for_node(tree.body[4], lines))


def test_is_iterable():
    def gen():
        yield

    class NoSpecialMethods(object):
        pass

    class HasIter(object):
        def __iter__(self):
            yield 1

    class HasGetItemAndLen(object):
        def __getitem__(self, i):
            return i ** 2

        def __len__(self):
            return 1 << 15

    class HasGetItem(object):
        def __getitem__(self, i):
            raise KeyError("tricked you, I am not iterable")

    class HasLen(object):
        def __len__(self, i):
            return -1

    assert is_iterable("")
    assert is_iterable([])
    assert is_iterable(range(1))
    assert is_iterable(gen())
    assert is_iterable({})
    assert is_iterable({}.keys())
    assert not is_iterable(42)
    assert not is_iterable(None)
    assert not is_iterable(False)
    assert not is_iterable(str)
    assert not is_iterable(NoSpecialMethods())
    assert is_iterable(HasIter())
    assert is_iterable(HasGetItemAndLen())
    assert not is_iterable(HasGetItem())
    assert not is_iterable(HasLen())
