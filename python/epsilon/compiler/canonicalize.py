"""Canonicalize problems into sum-of-prox form.

The basic building block is the prox_* functions each of which is a rule for
transforming an expression into one with an efficient proximal operator. These
support minor transformations but often times they simply serve to recognize
simple functions which have proximal operators.

The other major component is arbitrary fucntion composition through the epigraph
transformation which is handled in transform_expr_epigraph().
"""

from itertools import chain
import struct
import sys

from epsilon.expression import add
from epsilon.expression_pb2 import Expression, Problem, Curvature, Variable

# The prox_* functions recognize expressions with known proximal operators. The
# expressions returned need to match those defined in
# src/epsilon/operators/prox.cc
#
# TODO(mwytock): Add a test to verify tight coupling between python/C++ proximal
# operator forms by adding a python extension for HasProximalOperator() which
# calls into the C++ code and ensures we can evaluate a given prox.
def prox_affine(expr):
    if expr.curvature.curvature_type == Curvature.AFFINE:
        yield expr

def prox_least_squares(expr):
    if (expr.expression_type == Expression.POWER and
        expr.arg[0].expression_type == Expression.NORM_P and
        expr.p == 2 and expr.arg[0].p == 2 and
        expr.arg[0].arg[0].curvature.curvature_type == Curvature.AFFINE):
        yield expr

def prox_norm1(expr):
    if (expr.expression_type == Expression.NORM_P and
        expr.p == 1 and
        expr.arg[0].curvature.elementwise):
        yield expr

def prox_norm2(expr):
    if (expr.expression_type == Expression.NORM_P and
        expr.p == 2 and
        expr.arg[0].curvature.constant_multiple):
        yield expr

# First rule that matches takes precendence
PROX_RULES = [prox_least_squares,
              prox_norm1,
              prox_norm2,
              prox_affine]

def transform_expr_prox(expr):
    for prox_rule in PROX_RULES:
        prox_exprs = list(prox_rule(expr))
        if prox_exprs:
            return prox_exprs

def transform_expr_add(expr):
    if expr.expression_type == Expression.ADD:
        return expr.arg

def is_epigraph(expr):
    return (expr.expression_type == Expression.INDICATOR and
            expr.cone.cone_type == Cone.NON_ZERO and
            len(expr.arg) == 2 and
            expr.arg[0].expression_type == Expression.VARIABLE)

def epigraph_variable(expr):
    expr_str = struct.pack("q", hash(expr.SerializeToString())).encode("hex")
    var_id = ("epigraph:" + expr_str)

    return Expression(
        expression_type=Expression.VARIABLE,
        size=expr.size,
        variable=Variable(variable_id=var_id))

def transform_expr_epigraph(expr):
    # Epigraph form, I(f(g(x)) <= s) => I(f(t) <= s) + I(g(x) <= t)
    if is_epigraph(expr) and len(expr.arg[1].arg) == 1:
        f_expr = expr
        t_expr = epigraph_variable(expr.arg[1].arg[0])
        g_expr = epigraph(expr.arg[1].arg[0], t_expr)
        f.expr.arg[1].arg[0].CopyFrom(t_expr)
        return f_expr, g_expr

    # Standard form, f(g(x)) => f(t) + I(g(x) <= t)
    if len(expr.arg) == 1:
        f_expr = expr
        t_expr = epigraph_variable(expr.arg[0])
        g_expr = epigraph(expr.arg[0], t_expr)
        f_expr.arg[0].CopyFrom(t_expr)
        return f_expr, g_expr

    raise CanonicalizeError("unknown epigraph transformation")

def transform_expr(expr):
    prox_exprs = transform_expr_prox(expr)
    if prox_exprs:
        for prox_expr in prox_exprs:
            yield prox_expr
        return

    sub_exprs = transform_expr_add(expr)
    if sub_exprs:
        for prox_expr in chain(*(transform_expr(e) for e in sub_exprs)):
            yield prox_expr
        return

    f_expr, g_expr = transform_expr_epigraph(expr)
    for prox_expr in chain(transform_expr(f_expr), transform_expr(g_expr)):
        yield prox_expr


def transform(input):
    return Problem(objective=add(
        *chain(transform_expr(input.objective),
               *(transform_expr(constr) for constr in input.constraint))))
