"""Conic transforms for non-linear functions."""

import logging
from fractions import Fraction

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

def transform_max_elementwise(expr):
    t = epi_var(expr, "max_elementwise")
    return t, [expression.leq_constraint(x, t) for x in expr.arg]

def transform_max_entries(expr):
    t = epi_var(expr, "max_entries")
    return t, [expression.leq_constraint(x, t) for x in expr.arg]

def transform_min_elementwise(expr):
    t = epi_var(expr, "min_elementwise")
    return t, [expression.leq_constraint(t, x) for x in expr.arg]

def transform_quad_over_lin(expr):
    assert len(expr.arg) == 2
    x, y = expr.arg
    assert dim(y) == 1

    t = epi_var(expr, "qol", size=(1,1))
    return t, [
        expression.soc_constraint(
            expression.add(y, t),
            expression.vstack(
                expression.add(y, expression.negate(t)),
                expression.reshape(
                    expression.multiply(expression.scalar_constant(2), x),
                    dim(x), 1))),
        expression.leq_constraint(expression.scalar_constant(0), y)]

def transform_norm_p(expr):
    p = expr.p
    x = only_arg(expr)
    t = epi_var(expr, "norm_p", size=(1,1))

    if p == float("inf"):
        return t, [expression.leq_constraint(x, t),
                   expression.leq_constraint(expression.negate(x), t)]

    if p == 1:
        return transform_expr(expression.sum_entries(expression.abs_val(x)))

    if p == 2:
        return t, [expression.soc_constraint(t, x)]

    r = epi_var(expr, "norm_p_r", size=dims(x))
    p = Fraction(p)
    t1 = expression.multiply(expression.ones(*dims(x)), t)

    if p < 0:
        constrs = gm_constrs(t1, [x, r], (-p/(1-p), 1/(1-p)))
    elif 0 < p < 1:
        constrs = gm_constrs(r, [x, t1], (p, 1-p))
    elif p > 1:
        constrs = gm_constrs(x, [r, t1], (1/p, 1-1/p))

    constrs.append(expression.eq_constraint(expression.sum_entries(r), t))
    return t, constrs

def transform_norm_2_elementwise(expr):
    t = epi_var(expr, "norm_2_elementwise")
    return t, [expression.soc_elemwise_constraint(t, *expr.arg)]

def transform_power(expr):
    p = expr.p

    if p == 1:
        return only_arg(expr)

    one = expression.scalar_constant(1)
    if p == 0:
        return one, []

    t = epi_var(expr, "power")
    x = only_arg(expr)

    if p < 0:
        p, w = power_tools.pow_neg(p)
        constrs = gm_constrs(one, [x, t], w)
    if 0 < p < 1:
        p, w = power_tools.pow_mid(p)
        constrs = gm_constrs(t, [x, one], w)
    if p > 1:
        p, w = power_tools.pow_high(p)
        constrs = gm_constrs(x, [t, one], w)

    return t, constrs

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

def transform_geo_mean(expr):
    w = [Fraction(x.a, x.b) for x in expr.geo_mean_params.w]
    w_dyad = [Fraction(x.a, x.b) for x in expr.geo_mean_params.w_dyad]
    tree = power_tools.decompose(w_dyad)

    t = epi_var(expr, "geo_mean")
    x = only_arg(expr)
    x_list = [expression.index(x, i, i+1) for i in range(len(w))]
    return t, gm_constrs(t, x_list, w)

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
