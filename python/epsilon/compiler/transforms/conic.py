"""Conic transforms for non-linear functions."""

import logging

from epsilon import expression
from epsilon import tree_format
from epsilon import dcp
from epsilon.compiler.transforms.transform_util import *
from epsilon.expression_pb2 import Curvature, Expression

def transform_abs(expr):
    x = only_arg(expr)
    t = epi_var(expr, "abs")
    return t, [expression.leq_constraint(x, t),
               expression.leq_constraint(expression.negate(x), t)]

def transform_max_elemwise(expr):
    t = epi_var(expr, "max_elemwise")
    return t, [expression.leq_constraint(x, t) for x in args]

def transform_min_elemwise(expr):
    t = epi_var(expr, "min_elemwise")
    return t, [expression.leq_constraint(t, x) for x in args]

def transform_soc_elemwise(expr):
    t = epi_var(expr, "soc_elemwise")
    return t, [expression.soc_elemwise_constraint(t, x) for x in args]

def transform_quad_over_lin(expr):
    validate_args(expr, 2)
    x, y = expr.arg
    if dim(y) != 1:
        raise TransformError("quad_over_lin expects scalar y", expr)

    t = epi_var(expr, "qol", size=(1,1))
    return t, [
        expression.soc_constraint(
            expression.add(y, t),
            expression.vstack(
                expression.add(y, expression.negate(t)),
                expression.multiply(expression.constant(1, 1, scalar=2), x))),
        expression.leq_constraint(expression.constant(1, 1, scalar=0), y)]

def transform_norm_p(expr):
    if expr.p == 1:
        return transform_expr(
            expression.sum_entries(expression.abs_val(only_arg(expr))))

    raise TransformError("Unsupported p norm", expr)

def transform_expr(expr):
    logging.debug("conic transform_expr:\n%s", tree_format.format_expr(expr))

    constr = []
    for arg in expr.arg:
        arg_conic, arg_constr = transform_expr(arg)
        arg.CopyFrom(arg_conic)
        constr += arg_constr

    if not dcp.is_affine(expr):
        f_name = "transform_" + Expression.Type.Name(expr.expression_type).lower()
        if f_name not in globals():
            raise TransformError("No conic transform", expr)
        expr, expr_constr = globals()[f_name](expr)
        constr += expr_constr

    return expr, constr
