"""Compute DCP attributes of expressions.

This translates some of the functionality from cvxpy.atoms.atom.py and
cvxpy.expresison.expression.py to work on Epsilon's expression trees with the
help of cvxpy.utilities.
"""

import cvxpy.utilities

from epsilon import expression_pb2

def compute_sign(expr):
    return cvxpy.utilities.Sign(
        expression_pb2.Sign.Type.Name(expr.sign.sign_type))

def compute_shape(expr):
    return cvxpy.utilities.Shape(expr.size.dim[0], expr.size.dim[1])

def compute_curvature(expr):
    """Compute curvature based on DCP rules, from cvxpy.atoms.atom"""
    f_curvature = cvxpy.utilities.Curvature(
        expression_pb2.Curvature.Type.Name(expr.curvature.curvature_type))
    if not expr.arg:
        return f_curvature

    if expr.arg_monotonicity:
        ms = [expression_pb2.Monotonicity.Type.Name(m.monotonicity_type)
              for m in expr.arg_monotonicity]
        assert len(ms) == len(expr.arg)
    else:
        # Default
        ms = [cvxpy.utilities.monotonicity.NONMONOTONIC]*len(expr.arg)

    return reduce(
        lambda a, b: a+b,
        (cvxpy.utilities.monotonicity.dcp_curvature(
            monotonicity,
            f_curvature,
            get_dcp_attr(arg).sign,
            get_dcp_attr(arg).curvature)
         for arg, monotonicity in zip(expr.arg, ms)))

# TODO(mwytock): Should probably memoize this
def get_dcp_attr(expr):
    return cvxpy.utilities.DCPAttr(
        compute_sign(expr), compute_curvature(expr), compute_shape(expr))

def is_affine(expr):
    dcp_attr = get_dcp_attr(expr)
    return (dcp_attr.curvature == cvxpy.utilities.Curvature.AFFINE or
            dcp_attr.curvature == cvxpy.utilities.Curvature.CONSTANT)

def is_constant(expr):
    return get_dcp_attr(expr).curvature == cvxpy.utilities.Curvature.CONSTANT

def get_curvature(expr):
    dcp_attr = get_dcp_attr(expr)
    return expression_pb2.Curvature(
        curvature_type=expression_pb2.Curvature.Type.Value(
            dcp_attr.curvature.curvature_str))
