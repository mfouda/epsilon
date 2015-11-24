"""Compute DCP attributes of expressions.

This translates some of the functionality from cvxpy.atoms.atom.py and
cvxpy.expresison.expression.py to work on Epsilon's expression trees with the
help of cvxpy.utilities.
"""

import cvxpy.utilities

from epsilon import expression_pb2

class DCPProperties(object):
    def __init__(self, dcp_attr):
        self.dcp_attr = dcp_attr
        self.curvature = expression_pb2.Curvature(
            curvature_type=expression_pb2.Curvature.Type.Value(
                dcp_attr.curvature.curvature_str))

    @property
    def affine(self):
        return (self.dcp_attr.curvature == cvxpy.utilities.Curvature.AFFINE or
                self.dcp_attr.curvature == cvxpy.utilities.Curvature.CONSTANT)

    @property
    def constant(self):
        return self.dcp_attr.curvature == cvxpy.utilities.Curvature.CONSTANT

def compute_dcp_properties(expr):
    return DCPProperties(
        cvxpy.utilities.DCPAttr(
            compute_sign(expr), compute_curvature(expr), compute_shape(expr)))

def compute_sign(expr):
    return cvxpy.utilities.Sign(
        expression_pb2.Sign.Type.Name(expr.sign.sign_type))

def compute_shape(expr):
    return cvxpy.utilities.Shape(expr.size.dim[0], expr.size.dim[1])

def compute_curvature(expr):
    """Compute curvature based on DCP rules, from cvxpy.atoms.atom"""
    func_curvature = cvxpy.utilities.Curvature(
        expression_pb2.Curvature.Type.Name(expr.func_curvature.curvature_type))
    if not expr.arg:
        return func_curvature

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
            func_curvature,
            arg.dcp_props.dcp_attr.sign,
            arg.dcp_props.dcp_attr.curvature)
         for arg, monotonicity in zip(expr.arg, ms)))
