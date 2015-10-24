
from epsilon import tree_format

from nose.tools import assert_equal

def assert_expr_equal(a, b):
    # TODO(mwytock): Comparing the tree formats is kind of a "fuzzy equal". We
    # should do something better here.
    assert_equal(
        tree_format.format_expr(a), tree_format.format_expr(b),
        "\nExpected:\n" + tree_format.format_expr(a) +
        "\n!=\nActual:\n" + tree_format.format_expr(b))
