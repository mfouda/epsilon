"""Conic transforms for non-linear functions."""

import logging

from cvxpy.utilities import power_tools

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
    p = expr.p
    x = only_arg(expr)
    t = epi_var(expr, "norm_p", size=(1,1))
    if is_inf(p):
        return t, [expression.leq_constraint(x, t),
                   expression.leq_constraint(expression.negate(x), t)]

    if p == 1:
        return transform_expr(expression.sum_entries(expression.abs_val(x)))

    if p == 2:
        return t, [expression.soc_constraint(t, x)]

    # if p > 1:
    #     pass

    # if 0 < p < 1:
    #     pass

    # if p < 0:
    #     pass

    raise TransformError("Unsupported p norm", expr)

def transform_power(expr):
    p = expr.p

    if p == 1:
        return only_arg(expr)

    one = expression.promote(expression.scalar_constant(1), *dims(expr))
    if p == 0:
        return one, []

    t = epi_var(expr, "power")
    x = only_arg(expr)
    if p > 1:
        p, w = power_tools.pow_high(p)
        return t, gm_constrs(x, [t, one], w)

    if 0 < p < 1:
        p, w = power_tools.pow_mid(p)
        return t, gm_constrs(t, [x, one], w)

    if p < 0:
        p, w = power_tools.pow_neg(p)
        return t, gm_constrs(one, [x, t], w)

    raise TransformError("Unsupported power", expr)

def transform_huber(expr):
    n = epi_var(expr, "huber_n")
    s = epi_var(expr, "huber_s")

    # n**2 + 2*M*|s|
    t, constr = transform_expr(
        expression.add(
            expression.power(n, 2),
            expression.multiply(
                expression.scalar_constant(2*expr.M),
                expression.abs_val(s))))
    # x == s + n
    x = only_arg(expr)
    constr.append(expression.eq_constraint(x, expression.add(s, n)))
    return t, constr

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
