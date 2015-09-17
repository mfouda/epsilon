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

from epsilon.compiler import compiler_error
from epsilon.expression import *
from epsilon.expression_str import expr_str
from epsilon.expression_pb2 import Expression, Problem, Curvature, Variable

class CanonicalizeError(compiler_error.CompilerError):
    pass

def is_epigraph(expr):
    return (expr.expression_type == Expression.INDICATOR and
            expr.cone.cone_type == Cone.NON_NEGATIVE and
            len(expr.arg) == 2 and
            expr.arg[0].expression_type == Expression.VARIABLE)

def epigraph(f, t):
    """An expression for an epigraph constraint.

    The constraint depends on the curvature of f:
      - f convex,  I(f(x) <= t)
      - f concave, I(f(x) >= t)
      - f affine,  I(f(x) == t)
    """

    if f.curvature.curvature_type == Curvature.CONVEX:
        return indicator(Cone.NON_NEGATIVE, t, f)
    elif f.curvature.curvature_type == Curvature.CONCAVE:
        return indicator(Cone.NON_NEGATIVE, negate(t), negate(f))
    elif f.curvature.curvature_type == Curvature.AFFINE:
        return indicator(Cone.ZERO, t, f)
    else:
        raise CanonicalizeError("Unknown curvature", f)

def epigraph_variable(expr):
    expr_str = struct.pack("q", hash(expr.SerializeToString())).encode("hex")
    m, n = expr.size.dim
    return variable(m, n, "canonicalize:" + expr_str)

def transform_epigraph(f_expr, g_expr):
    t_expr = epigraph_variable(g_expr)
    epi_g_expr = epigraph(g_expr, t_expr)
    g_expr.CopyFrom(t_expr)

    yield f_expr
    for prox_expr in transform_expr(epi_g_expr):
        yield prox_expr

# The prox_* functions recognize expressions with known proximal operators. The
# expressions returned need to match those defined in
# src/epsilon/operators/prox.cc

# General rules for dealing with additional and scalar multiplication
def prox_multiply_scalar(expr):
    if (expr.expression_type == Expression.MULTIPLY and
        expr.arg[0].curvature.curvature_type == Curvature.CONSTANT and
        dimension(expr.arg[0]) == 1):
        for prox_expr in transform_expr(expr.arg[1]):
            if prox_expr.expression_type == Expression.INDICATOR:
                yield prox_expr
            else:
                yield multiply(expr.arg[0], prox_expr)

def prox_add(expr):
    if expr.expression_type == Expression.ADD:
        for arg in expr.arg:
            for prox_expr in transform_expr(arg):
                yield prox_expr

# Rules for known proximal operators
def prox_least_squares(expr):
    if (expr.expression_type == Expression.POWER and
        expr.arg[0].expression_type == Expression.NORM_P and
        expr.p == 2 and expr.arg[0].p == 2):
        expr.proximal_operator.name = "LeastSquaresProx"
        if expr.arg[0].arg[0].curvature.curvature_type == Curvature.AFFINE:
            yield expr
        else:
            raise NotImplementedError()

def prox_norm1(expr):
    if (expr.expression_type == Expression.NORM_P and expr.p == 1):
        expr.proximal_operator.name = "NormL1Prox"
        if expr.arg[0].curvature.elementwise:
            yield expr
        else:
            raise NotImplementedError()

def prox_norm2(expr):
    if (expr.expression_type == Expression.NORM_P and expr.p == 2):
        expr.proximal_operator.name = "NormL2Prox"
        if expr.arg[0].curvature.scalar_multiple:
            yield expr
        else:
            raise NotImplementedError()

def prox_exp(expr):
    if expr.expression_type == Expression.EXP:
        expr.proximal_operator.name = "ExpProx"
        if expr.arg[0].curvature.elementwise:
            yield expr
        else:
            for prox_expr in transform_epigraph(expr, expr.arg[0]):
                yield prox_expr

def prox_equality_constraint(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.ZERO and
        all(arg.curvature.curvature_type == Curvature.AFFINE or
            arg.curvature.curvature_type == Curvature.CONSTANT
            for arg in expr.arg)):
        expr.proximal_operator.name = "EqualityConstraintProx"
        yield expr

def prox_affine(expr):
    if (expr.curvature.curvature_type == Curvature.CONSTANT or
        expr.curvature.curvature_type == Curvature.AFFINE):
        expr.proximal_operator.name = "AffineProx"
        yield expr

# Matrix rules, f(alpha*X)
def prox_norm12(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.NORM_2_ELEMENTWISE):

        # Rewrite this as l1/l2 norm using reshape() and hstack()
        m = dimension(expr.arg[0].arg[0])
        arg = hstack(*(reshape(arg, m, 1) for arg in expr.arg[0].arg))
        expr = norm_pq(arg, 1, 2)

        expr.proximal_operator.name = "NormL1L2Prox"
        if arg.curvature.scalar_multiple:
            yield expr
        else:
            for prox_expr in transform_epigraph(expr, expr.arg[0]):
                yield prox_expr

def prox_neg_log_det(expr):
    if (expr.expression_type == Expression.NEGATE and
        expr.arg[0].expression_type == Expression.LOG_DET):

        expr.proximal_operator.name = "NegativeLogDetProx"
        if expr.arg[0].arg[0].curvature.scalar_multiple:
            yield expr
        else:
            raise NotImplementedError()

def prox_epigraph(expr):
    if is_epigraph(expr):
        t_expr, f_expr = expr.arg

        # The basic algorithm is to canonicalize the f(x) expression into f_1(x)
        # + ... f_n(x) using the transform_expr(). For each f_i there are two
        # cases to consider:
        #
        # 1) If f_i is an indicator function, we emit it
        # 2) Otherwise, we emit a new epigraph constraint I(f_i(x) <= t_i) and
        # keep track of t_i
        #
        # Finally, we emit I(t_1 + ... + t_n == t)
        #
        # NOTE(mwytock): This makes the assumption that we have a proximal
        # operator for I(f_i(x) <= t) if we have one for f_i(x)
        ti_exprs = []
        for fi_expr in transform_expr(f_expr):
            if fi_expr.expression_type == Expression.INDICATOR:
                yield fi_expr
            else:
                ti_expr = epigraph_variable(fi_expr)
                ti_exprs.append(ti_expr)
                yield epigraph(fi_expr, ti_expr)

        yield equality_constraint(add(*ti_exprs), t_expr)

# NOTE(mwtyock): first rule that matches takes precendence
PROX_RULES = [f for name, f in locals().items() if name.startswith("prox_")]

def transform_expr(expr):
    for prox_rule in PROX_RULES:
        prox_exprs = list(prox_rule(expr))
        if prox_exprs:
            return prox_exprs

    raise CanonicalizeError("No prox rule", expr)

def transform(input):
    prox_exprs = list(transform_expr(input.objective))
    for constr in input.constraint:
        prox_exprs += list(transform_expr(constr))
    return Problem(objective=add(*prox_exprs))
