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

from epsilon import error
from epsilon.expression import *
from epsilon.expression_pb2 import Expression, Problem, Curvature, Variable
from epsilon.expression_str import expr_str

class CanonicalizeError(error.ExpressionError):
    pass

def is_epigraph(expr):
    return (expr.expression_type == Expression.INDICATOR and
            expr.cone.cone_type == Cone.NON_NEGATIVE and
            len(expr.arg) == 2 and
            expr.arg[0].expression_type == Expression.VARIABLE)

LINEAR_EXPRESSION_TYPES = set([
    Expression.INDEX,
    Expression.NEGATE,
    Expression.SUM,
    Expression.TRANSPOSE,
    Expression.HSTACK,
    Expression.VSTACK,
    Expression.TRACE,
    Expression.RESHAPE])

def epigraph(f, t):
    """An expression for an epigraph constraint.

    The constraint depends on the curvature of f:
      - f convex,  I(f(x) <= t)
      - f concave, I(f(x) >= t)
      - f affine,  I(f(x) == t)
    """

    if f.curvature.curvature_type == Curvature.CONVEX:
        return leq_constraint(f, t)
    elif f.curvature.curvature_type == Curvature.CONCAVE:
        return leq_constraint(negate(f), negate(t))
    elif f.curvature.curvature_type == Curvature.AFFINE:
        return equality_constraint(f, t);
    else:
        raise CanonicalizeError("Unknown curvature", f)

def fp_expr(expr):
    return struct.pack("q", hash(expr.SerializeToString())).encode("hex")

def epigraph_variable(expr):
    m, n = expr.size.dim
    return variable(m, n, "canonicalize:" + fp_expr(expr))

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
def prox_affine(expr):
    if (expr.curvature.curvature_type == Curvature.CONSTANT or
        expr.curvature.curvature_type == Curvature.AFFINE):
        expr.proximal_operator.name = "AffineProx"
        yield expr

def prox_fused_lasso(expr):
    # TODO(mwytock): Make this more flexible? Support weighted form?
    # TODO(mwytock): Rewrite this using a new expression type for TV(x)
    if (expr.expression_type == Expression.NORM_P and expr.p == 1 and
        expr.arg[0].expression_type == Expression.ADD and
        expr.arg[0].arg[0].expression_type == Expression.INDEX and
        expr.arg[0].arg[0].arg[0].expression_type == Expression.VARIABLE and
        expr.arg[0].arg[1].expression_type == Expression.NEGATE and
        expr.arg[0].arg[1].arg[0].expression_type == Expression.INDEX and
        expr.arg[0].arg[1].arg[0].arg[0].expression_type ==
        Expression.VARIABLE):

        var_id0 = expr.arg[0].arg[0].arg[0].variable.variable_id
        var_id1 = expr.arg[0].arg[1].arg[0].arg[0].variable.variable_id
        if var_id0 != var_id1:
            return

        expr.proximal_operator.name = "FusedLassoProx"
        yield expr

def prox_least_squares(expr):
    if (expr.expression_type == Expression.QUAD_OVER_LIN and
        expr.arg[1].expression_type == Expression.CONSTANT and
        expr.arg[1].constant.scalar == 1):
        arg = expr.arg[0]
    elif ((expr.expression_type == Expression.POWER and
           expr.arg[0].expression_type == Expression.NORM_P and
           expr.p == 2 and expr.arg[0].p == 2) or
          (expr.expression_type == Expression.SUM and
           expr.arg[0].expression_type == Expression.POWER and
           expr.arg[0].p == 2)):
        arg = expr.arg[0].arg[0]

    else:
        return

    expr = sum_entries(power(arg, 2))
    m, n = arg.size.dim
    expr.proximal_operator.name = (
        "LeastSquaresMatrixProx" if n > 1 else "LeastSquaresProx")

    if expr.arg[0].arg[0].curvature.curvature_type == Curvature.AFFINE:
        yield expr
    else:
        for prox_expr in transform_epigraph(expr, expr.arg[0].arg[0]):
            yield prox_expr

def prox_logistic(expr):
    if (expr.expression_type == Expression.LOG_SUM_EXP and
        expr.arg[0].expression_type == Expression.VSTACK and
        len(expr.arg[0].arg) == 2 and
        expr.arg[0].arg[0].expression_type == Expression.CONSTANT):

        expr.proximal_operator.name = "LogisticProx"
        if expr.arg[0].arg[1].curvature.elementwise:
            yield expr
        else:
            for prox_expr in transform_epigraph(expr, expr.arg[0].arg[1]):
                yield prox_expr

def prox_norm1(expr):
    if (expr.expression_type == Expression.NORM_P and expr.p == 1):
        expr.proximal_operator.name = "NormL1Prox"
        if expr.arg[0].curvature.elementwise:
            yield expr
        else:
            for prox_expr in transform_epigraph(expr, expr.arg[0]):
                yield prox_expr

def prox_norm2(expr):
    if (expr.expression_type == Expression.NORM_P and expr.p == 2):
        expr.proximal_operator.name = "NormL2Prox"
        if expr.arg[0].curvature.scalar_multiple:
            yield expr
        else:
            for prox_expr in transform_epigraph(expr, expr.arg[0]):
                yield prox_expr

def prox_exp(expr):
    if expr.expression_type == Expression.EXP:
        expr.proximal_operator.name = "ExpProx"
        if expr.arg[0].curvature.elementwise:
            yield expr
        else:
            for prox_expr in transform_epigraph(expr, expr.arg[0]):
                yield prox_expr

def prox_huber(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.HUBER):

        # Represent huber function as
        # minimize   n^2 + 2M|s|
        # subject to s + n = x
        arg = expr.arg[0].arg[0]
        m, n = arg.size.dim
        square_var = variable(m, n, "canonicalize:huber_square:" + fp_expr(arg))
        abs_var = variable(m, n, "canonicalize:huber_abs:" + fp_expr(arg))

        exprs = [
            equality_constraint(add(square_var, abs_var), arg),
            sum_entries(power(square_var, 2)),
            multiply(constant(1, 1, 2*expr.arg[0].M), norm_p(abs_var, 1))]

        for expr in exprs:
            for prox_expr in transform_expr(expr):
                yield prox_expr

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

def prox_negative_log(expr):
    if (expr.expression_type == Expression.NEGATE and
        expr.arg[0].expression_type == Expression.SUM and
        expr.arg[0].arg[0].expression_type == Expression.LOG):

        expr.proximal_operator.name = "NegativeLogProx"
        if expr.arg[0].arg[0].arg[0].curvature.scalar_multiple:
            yield expr
        else:
            raise NotImplementedError()

def prox_negative_entropy(expr):
    if (expr.expression_type == Expression.NEGATE and
        expr.arg[0].expression_type == Expression.SUM and
        expr.arg[0].arg[0].expression_type == Expression.ENTR):

        expr.proximal_operator.name = "NegativeEntropyProx"
        if expr.arg[0].arg[0].arg[0].curvature.scalar_multiple:
            yield expr
        else:
            raise NotImplementedError()

# Piecewise Linear Family
def is_hinge(expr):
    return (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.MAX_ELEMENTWISE and
        expr.arg[0].arg[0].expression_type == Expression.ADD and
        expr.arg[0].arg[0].arg[0].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[0].arg[0].constant.scalar == 1. and
        expr.arg[0].arg[0].arg[1].expression_type == Expression.NEGATE and
        expr.arg[0].arg[0].arg[1].arg[0].expression_type == Expression.VARIABLE and
        expr.arg[0].arg[1].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[1].constant.scalar == 0
        )

def is_norm_l1_asymetric(expr):
    return (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.ADD and
        expr.arg[0].arg[0].expression_type == Expression.MULTIPLY and
        expr.arg[0].arg[0].arg[0].expression_type == Expression.MAX_ELEMENTWISE and
        expr.arg[0].arg[0].arg[0].arg[0].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[0].arg[0].arg[0].constant.scalar == 0 and
        expr.arg[0].arg[0].arg[0].arg[1].expression_type == Expression.VARIABLE and
        expr.arg[0].arg[0].arg[1].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[0].arg[1].constant.scalar >= 0 and
        expr.arg[1].arg[0].expression_type == Expression.MULTIPLY and
        expr.arg[1].arg[0].arg[0].expression_type == Expression.MAX_ELEMENTWISE and
        expr.arg[1].arg[0].arg[0].arg[0].expression_type == Expression.CONSTANT and
        expr.arg[1].arg[0].arg[0].arg[0].constant.scalar == 0 and
        expr.arg[1].arg[0].arg[0].arg[1].expression_type == Expression.NEGATE and
        expr.arg[1].arg[0].arg[0].arg[1].arg[0].expression_type == Expression.VARIABLE and
        expr.arg[1].arg[0].arg[1].expression_type == Expression.CONSTANT and
        expr.arg[1].arg[0].arg[1].constant.scalar >= 0
        )

def is_deadzone(expr):
    return (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.ADD and
        expr.arg[0].arg[0].expression_type == Expression.MAX_ELEMENTWISE and
        expr.arg[0].arg[0].arg[0].expression_type == Expression.ADD and
        expr.arg[0].arg[0].arg[0].arg[0].expression_type == Expression.VARIABLE and
        expr.arg[0].arg[0].arg[0].arg[1].expression_type == Expression.NEGATE and
        expr.arg[0].arg[0].arg[0].arg[1].arg[0].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[0].arg[1].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[0].arg[1].constant.scalar == 0 and
        expr.arg[0].arg[1].expression_type == Expression.MAX_ELEMENTWISE and
        expr.arg[0].arg[1].arg[0].expression_type == Expression.ADD and
        expr.arg[0].arg[1].arg[0].arg[0].expression_type == Expression.NEGATE and
        expr.arg[0].arg[1].arg[0].arg[0].arg[0].expression_type == Expression.VARIABLE and
        expr.arg[0].arg[1].arg[0].arg[1].expression_type == Expression.NEGATE and
        expr.arg[0].arg[1].arg[0].arg[1].arg[0].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[1].arg[1].constant.scalar == 0 
        )

def prox_hinge(expr):
    if is_hinge(expr):
        expr.proximal_operator.name = "HingeProx"
        yield expr

def prox_norm_l1_asymetric(expr):
    if is_norm_l1_asymetric(expr):
        expr.proximal_operator.name = "NormL1AsymetricProx"
        yield expr

def prox_deadzone(expr):
    if is_deadzone(expr):
        expr.proximal_operator.name = "DeadZoneProx"
        yield expr

def prox_max_elementwise(expr):
    """Replace max{..., ...} with epigraph constraints"""
    if expr.expression_type != Expression.MAX_ELEMENTWISE:
        return

    m, n = expr.size.dim
    t = variable(m, n, "canonicalize:max:" + fp_expr(expr))
    yield t
    for arg in expr.arg:
        for prox_expr in transform_expr(leq_constraint(arg, t)):
            yield prox_expr

def prox_linear_epigraph(expr):
    if not expr.expression_type in LINEAR_EXPRESSION_TYPES:
        return

    expr.proximal_operator.name = "AffineProx"
    new_args = []
    for arg in expr.arg:
        for prox_expr in transform_expr(arg):
            if prox_expr.expression_type == Expression.INDICATOR:
                yield prox_expr
            else:
                new_args.append(prox_expr)

    expr.ClearField("arg")
    for arg in new_args:
        expr.arg.add().CopyFrom(arg)

    expr.curvature.curvature_type = Curvature.AFFINE
    yield expr

# Rules for epigraph forms
def prox_norm2_epigraph(expr):
    if (is_epigraph(expr) and
        expr.arg[1].expression_type == Expression.NORM_P and
        expr.arg[1].p == 2):

        expr.proximal_operator.name = "NormL2Epigraph"
        if expr.arg[1].arg[0].curvature.scalar_multiple:
            yield expr

def prox_norm1_epigraph(expr):
    if (is_epigraph(expr) and
        expr.arg[1].expression_type == Expression.NORM_P and
        expr.arg[1].p == 1):

        expr.proximal_operator.name = "NormL1Epigraph"
        if expr.arg[1].arg[0].curvature.scalar_multiple:
            yield expr

def prox_negative_log_epigraph(expr):
    if (is_epigraph(expr) and
        expr.arg[1].expression_type == Expression.NEGATE and
        expr.arg[1].arg[0].expression_type == Expression.SUM and
        expr.arg[1].arg[0].arg[0].expression_type == Expression.LOG):

        expr.proximal_operator.name = "NegativeLogEpigraph"
        if expr.arg[1].arg[0].arg[0].arg[0].curvature.scalar_multiple:
            yield expr

def prox_negative_entropy_epigraph(expr):
    if (is_epigraph(expr) and
        expr.arg[1].expression_type == Expression.NEGATE and
        expr.arg[1].arg[0].expression_type == Expression.SUM and
        expr.arg[1].arg[0].arg[0].expression_type == Expression.ENTR):

        expr.proximal_operator.name = "NegativeEntropyEpigraph"
        if expr.arg[1].arg[0].arg[0].arg[0].curvature.scalar_multiple:
            yield expr

def prox_logistic_epigraph(expr):
    if (is_epigraph(expr) and
        expr.arg[1].expression_type == Expression.LOG_SUM_EXP and
        expr.arg[1].arg[0].expression_type == Expression.VSTACK and
        len(expr.arg[1].arg[0].arg) == 2 and
        expr.arg[1].arg[0].arg[0].expression_type == Expression.CONSTANT):

        expr.proximal_operator.name = "LogisticEpigraph"
        if expr.arg[1].arg[0].arg[1].curvature.elementwise:
            yield expr

def prox_hinge_epigraph(expr):
    if is_epigraph(expr) and is_hinge(expr.arg[1]):
        expr.proximal_operator.name = "HingeEpigraph"
        yield expr

def prox_equality_constraint(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.ZERO):

        expr.proximal_operator.name = "LinearEqualityProx"
        if all(arg.curvature.curvature_type == Curvature.AFFINE or
               arg.curvature.curvature_type == Curvature.CONSTANT
               for arg in expr.arg):
            yield expr

def prox_non_negative(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.NON_NEGATIVE):

        expr.proximal_operator.name = "NonNegativeProx"
        if all(arg.curvature.scalar_multiple for arg in expr.arg):
            yield expr
        elif all(arg.curvature.curvature_type == Curvature.AFFINE or
                 arg.curvature.curvature_type == Curvature.CONSTANT
                 for arg in expr.arg):

            add_expr = add(expr.arg[0], *(negate(arg) for arg in expr.arg[1:]))
            m, n = add_expr.size.dim
            y = variable(m, n, "canonicalize:non_negative:" + fp_expr(add_expr))

            exprs = [non_negative(y), equality_constraint(y, add_expr)]
            for expr in exprs:
                for prox_expr in transform_expr(expr):
                    yield prox_expr

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

# NOTE(mwytock): This sorts rules by order in which theyre defined
PROX_RULES = [f for name, f in locals().items() if name.startswith("prox_")]
PROX_RULES.sort()

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
