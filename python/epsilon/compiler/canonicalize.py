"""Canonicalize problems into sum-of-prox form.

The basic building block is the prox_* functions each of which is a rule for
transforming an expression into one with an efficient proximal operator. These
support minor transformations but often times they simply serve to recognize
simple functions which have proximal operators.

The other major component is arbitrary fucntion composition through the epigraph
transformation which is handled in transform_expr_epigraph().
"""


from collections import defaultdict

from epsilon import error
from epsilon import expression
from epsilon.expression import *
from epsilon.expression_pb2 import Expression, Problem, Curvature, Variable
from epsilon.expression_util import *

class CanonicalizeError(error.ExpressionError):
    pass

def is_epigraph(expr):
    """Epigraph represented with expression I(t - f(x) >= 0)."""
    return (expr.expression_type == Expression.INDICATOR and
            expr.cone.cone_type == Cone.NON_NEGATIVE and
            len(expr.arg) == 1 and
            len(expr.arg[0].arg) == 2 and
            len(expr.arg[0].arg[1].arg) == 1 and
            expr.arg[0].expression_type == Expression.ADD and
            expr.arg[0].arg[0].expression_type == Expression.VARIABLE and
            expr.arg[0].arg[1].expression_type == Expression.NEGATE)

def get_epigraph(expr):
    assert is_epigraph(expr)
    t_expr = expr.arg[0].arg[0]
    f_expr = expr.arg[0].arg[1].arg[0]
    return f_expr, t_expr

def get_scalar_multiply(expr):
    if not (expr.expression_type == Expression.MULTIPLY and len(expr.arg) == 2):
        return False

    for i, arg in enumerate(expr.arg):
        if arg.curvature.curvature_type == Curvature.CONSTANT and dim(arg) == 1:
            if arg.expression_type != Expression.CONSTANT:
                raise CanonicalizeError("unexpected scalar constant", arg)
            return arg.constant.scalar, expr.arg[1-i]

    return False

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

def epigraph_variable(expr):
    m, n = expr.size.dim
    return variable(m, n, "canonicalize:" + fp_expr(expr))

def transform_epigraph(f_expr, g_expr):
    t_expr = epigraph_variable(g_expr)
    epi_g_expr = epigraph(g_expr, t_expr)
    g_expr.CopyFrom(t_expr)

    for prox_expr in transform_expr(epi_g_expr):
        yield prox_expr
    yield f_expr

# Piecewise Linear Family
# TODO(mwytock): Replace these with a single ScaledZone function which is what
# they are mapped to anyways.
def is_hinge(expr):
    return (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.MAX_ELEMENTWISE and
        expr.arg[0].arg[0].expression_type == Expression.ADD and
        expr.arg[0].arg[0].arg[0].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[0].arg[0].constant.scalar == 1. and
        expr.arg[0].arg[0].arg[1].expression_type == Expression.NEGATE and
        expr.arg[0].arg[1].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[1].constant.scalar == 0)

def is_norm_l1_asymmetric(expr):
    return (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.ADD and
        expr.arg[0].arg[0].expression_type == Expression.MULTIPLY and
        expr.arg[0].arg[0].arg[0].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[0].arg[0].constant.scalar >= 0 and
        expr.arg[0].arg[0].arg[1].expression_type == Expression.MAX_ELEMENTWISE and
        expr.arg[0].arg[0].arg[1].arg[0].expression_type == Expression.VARIABLE and
        expr.arg[0].arg[0].arg[1].arg[1].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[0].arg[1].arg[1].constant.scalar == 0 and
        expr.arg[0].arg[1].expression_type == Expression.MULTIPLY and
        expr.arg[0].arg[1].arg[0].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[1].arg[0].constant.scalar >= 0 and
        expr.arg[0].arg[1].arg[1].expression_type == Expression.MAX_ELEMENTWISE and
        expr.arg[0].arg[1].arg[1].arg[0].expression_type == Expression.NEGATE and
        expr.arg[0].arg[1].arg[1].arg[0].arg[0].expression_type == Expression.VARIABLE and
        expr.arg[0].arg[1].arg[1].arg[1].expression_type == Expression.CONSTANT and
        expr.arg[0].arg[1].arg[1].arg[1].constant.scalar == 0
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

def is_inv_pos(expr):
    return (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.POWER and
        expr.arg[0].p == -1 and
        expr.arg[0].arg[0].expression_type == Expression.VARIABLE)

# TODO(mwytock): Currently these are only used for the epigraph form, use them
# for proximal operators in the objective as well.
EXPRESSION_RULES = [
    ("DeadZone", Expression.SUM, is_deadzone),
    ("SumExp", Expression.SUM,
     lambda e: (e.arg[0].expression_type == Expression.EXP)),
    ("InvPos", Expression.SUM, is_inv_pos),
    ("Hinge", Expression.SUM, is_hinge),
    ("LambdaMax", Expression.LAMBDA_MAX,
     lambda e: (e.arg[0].expression_type == Expression.VARIABLE)),
    ("Logistic", Expression.SUM,
     lambda e: (e.arg[0].expression_type == Expression.LOGISTIC)),
    ("KLDiv", Expression.KL_DIV,
     lambda e: (e.arg[0].expression_type == Expression.VARIABLE)),
    ("MaxEntries", Expression.MAX_ENTRIES,
     lambda e: (e.arg[0].expression_type == Expression.VARIABLE)),
    ("NegativeEntropy", Expression.NEGATE,
     lambda e: (e.arg[0].expression_type == Expression.SUM and
                e.arg[0].arg[0].expression_type == Expression.ENTR and
                e.arg[0].arg[0].arg[0].curvature.elementwise)),
    ("NegativeLog", Expression.NEGATE,
     lambda e: (e.arg[0].expression_type == Expression.SUM and
                e.arg[0].arg[0].expression_type == Expression.LOG and
                e.arg[0].arg[0].arg[0].curvature.elementwise)),
    ("NegativeLogDet", Expression.NEGATE,
     lambda e: (e.arg[0].expression_type == Expression.LOG_DET and
                e.arg[0].arg[0].curvature.scalar_multiple)),
    ("NormL1", Expression.NORM_P,
     lambda e: (e.p == 1 and e.arg[0].curvature.scalar_multiple)),
    ("NormL1Asymmetric", Expression.SUM, is_norm_l1_asymmetric),
    ("NormL2", Expression.NORM_P,
     lambda e: (e.p == 2 and e.arg[0].curvature.scalar_multiple)),
    ("NormNuclear", Expression.NORM_NUC,
     lambda e: (e.arg[0].curvature.scalar_multiple)),
]

EXPRESSION_RULE_MAP = defaultdict(list)
for name, expression_type, rule in EXPRESSION_RULES:
    EXPRESSION_RULE_MAP[expression_type].append((rule, name))

def get_function(expr):
    for rule, name in EXPRESSION_RULE_MAP[expr.expression_type]:
        if rule(expr):
            return name

# The prox_* functions recognize expressions with known proximal operators. The
# expressions returned need to match those defined in
# src/epsilon/operators/prox.cc

# General rules for dealing with additional and scalar multiplication
def prox_scalar_multiply(expr):
    args = get_scalar_multiply(expr)
    if not args:
        return

    alpha, expr = args
    for prox_expr in transform_expr(expr):
        if prox_expr.expression_type == Expression.INDICATOR:
            yield prox_expr
        else:
            prox_args = get_scalar_multiply(prox_expr)
            if prox_args:
                beta, prox_expr = prox_args
                if alpha*beta == 1:
                    yield prox_expr
                else:
                    yield expression.multiply(
                        expression.scalar_constant(alpha*beta), prox_expr)
            else:
                yield expression.multiply(
                    expression.scalar_constant(alpha), prox_expr)

def prox_negate(expr):
    if expr.expression_type != Expression.NEGATE:
        return

    negate_arg = expression.multiply(
        expression.scalar_constant(-1), only_arg(expr))
    for prox_expr in transform_expr(negate_arg):
        yield prox_expr

def prox_add(expr):
    if expr.expression_type == Expression.ADD:
        for arg in expr.arg:
            for prox_expr in transform_expr(arg):
                yield prox_expr

# Rules for known proximal operators
def prox_affine(expr):
    if (expr.curvature.curvature_type == Curvature.CONSTANT or
        expr.curvature.curvature_type == Curvature.AFFINE):
        expr.proximal_operator.name = "LinearProx"
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

def prox_kl_div(expr):
    if (expr.expression_type == Expression.KL_DIV and
        expr.arg[0].expression_type == Expression.VARIABLE):
        expr.proximal_operator.name = "KLDivProx"
        yield expr

def prox_lambda_max(expr):
    if (expr.expression_type == Expression.LAMBDA_MAX and
        expr.arg[0].expression_type == Expression.VARIABLE):
        expr.proximal_operator.name = "LambdaMaxProx"
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
    expr.proximal_operator.name = "LeastSquaresProx"

    if expr.arg[0].arg[0].curvature.curvature_type == Curvature.AFFINE:
        yield expr
    else:
        for prox_expr in transform_epigraph(expr, expr.arg[0].arg[0]):
            yield prox_expr

def prox_logistic(expr):
    if (expr.expression_type == Expression.SUM and
        expr.arg[0].expression_type == Expression.LOGISTIC):

        expr.proximal_operator.name = "LogisticProx"
        if expr.arg[0].arg[0].curvature.elementwise:
            yield expr
        else:
            for prox_expr in transform_epigraph(expr, expr.arg[0].arg[0]):
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

def prox_norm_nuc(expr):
    if (expr.expression_type == Expression.NORM_NUC):
        expr.proximal_operator.name = "NormNuclearProx"
        if expr.arg[0].curvature.scalar_multiple:
            yield expr
        else:
            for prox_expr in transform_epigraph(expr, expr.arg[0]):
                yield prox_expr

def prox_sum_exp(expr):
    if expr.expression_type == Expression.EXP:
        expr.proximal_operator.name = "SumExpProx"
        if expr.arg[0].curvature.elementwise:
            yield expr
        else:
            for prox_expr in transform_epigraph(expr, expr.arg[0]):
                yield prox_expr
    elif (expr.expression_type == Expression.SUM and
          expr.arg[0].expression_type == Expression.EXP and
          expr.arg[0].arg[0].expression_type == Expression.VARIABLE):
        expr.proximal_operator.name = "SumExpProx"
        yield expr

def prox_sum_largest(expr):
    if (expr.expression_type == Expression.SUM_LARGEST and
        expr.arg[0].expression_type == Expression.VARIABLE):
        expr = sum_largest(expr.arg[0], expr.k)
        expr.proximal_operator.name = "SumLargestProx"
        yield expr

def prox_inv_pos(expr):
    if is_inv_pos(expr):
        expr.proximal_operator.name = "InvPosProx"
        yield expr

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
        m = dim(expr.arg[0].arg[0])
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

def prox_hinge(expr):
    if is_hinge(expr):
        arg = expr.arg[0].arg[0].arg[1].arg[0]
        expr.proximal_operator.name = "HingeProx"
        if arg.curvature.scalar_multiple:
            yield expr
        else:
            for prox_expr in transform_epigraph(expr, arg):
                yield prox_expr

def prox_norm_l1_asymmetric(expr):
    if is_norm_l1_asymmetric(expr):
        expr.proximal_operator.name = "NormL1AsymmetricProx"
        yield expr

def prox_deadzone(expr):
    if is_deadzone(expr):
        expr.proximal_operator.name = "DeadZoneProx"
        yield expr

def prox_norm_l1_asymmetric_single_max(expr):
    if not (expr.expression_type == Expression.SUM and
            expr.arg[0].expression_type == Expression.MAX_ELEMENTWISE and
            len(expr.arg[0].arg) == 2 and
            expr.arg[0].arg[0].expression_type == Expression.MULTIPLY and
            expr.arg[0].arg[1].expression_type == Expression.MULTIPLY):
        return

    arg0 = expr.arg[0].arg[0].arg[1]
    arg1 = expr.arg[0].arg[1].arg[1]
    if arg0 != arg1:
        return

    alpha = expr.arg[0].arg[0].arg[0].constant.scalar
    beta = expr.arg[0].arg[1].arg[0].constant.scalar
    if alpha < 0:
        tmp = beta
        beta = -alpha
        alpha = tmp
    else:
        assert beta < 0
        beta = -beta

    expr = scaled_zone(arg0, alpha, beta, 0, 0)
    expr.proximal_operator.name = "ScaledZoneProx"

    arg = expr.arg[0]
    if arg.curvature.elementwise:
        yield expr
    else:
        for prox_expr in transform_epigraph(expr, arg):
            yield prox_expr

def prox_matrix_frac(expr):
    if expr.expression_type == Expression.MATRIX_FRAC:
        expr.proximal_operator.name = 'MatrixFracProx'
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

def prox_max_entries(expr):
    if (expr.expression_type == Expression.MAX_ENTRIES and
        expr.arg[0].expression_type == Expression.VARIABLE):
        expr.proximal_operator.name = "MaxEntriesProx"
        yield expr

def prox_epigraph_atomic(expr):
    if not is_epigraph(expr):
        return

    f_expr, t_expr = get_epigraph(expr)
    name = get_function(f_expr)
    if name:
        expr.proximal_operator.name = name + "Epigraph"
        yield expr

def prox_linear_equality(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type == Cone.ZERO and
        len(expr.arg) == 1 and
        expr.arg[0].curvature.curvature_type == Curvature.AFFINE):
        expr.proximal_operator.name = "LinearEqualityProx"
        yield expr


CONE_PROX = {
    Cone.NON_NEGATIVE: "NonNegativeProx",
    Cone.SEMIDEFINITE: "SemidefiniteProx",
}

def prox_cone(expr):
    if (expr.expression_type == Expression.INDICATOR and
        expr.cone.cone_type in CONE_PROX):

        expr.proximal_operator.name = CONE_PROX[expr.cone.cone_type]

        if all(arg.curvature.scalar_multiple for arg in expr.arg):
            yield expr
        elif all(arg.curvature.curvature_type == Curvature.AFFINE or
                 arg.curvature.curvature_type == Curvature.CONSTANT
                 for arg in expr.arg):

            add_expr = add(expr.arg[0], *(negate(arg) for arg in expr.arg[1:]))
            m, n = add_expr.size.dim
            y = variable(m, n, "canonicalize:cone:" + fp_expr(add_expr))

            exprs = [indicator(expr.cone.cone_type, y),
                     equality_constraint(y, add_expr)]
            for expr in exprs:
                for prox_expr in transform_expr(expr):
                    yield prox_expr

def prox_linear_epigraph(expr):
    if not expr.expression_type in LINEAR_EXPRESSION_TYPES:
        return

    expr.proximal_operator.name = "LinearProx"
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

def prox_epigraph(expr):
    if is_epigraph(expr):
        f_expr, t_expr = get_epigraph(expr)

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

PROX_RULES = [
    # Prox rules
    prox_affine,
    prox_fused_lasso,
    prox_kl_div,
    prox_lambda_max,
    prox_least_squares,
    prox_logistic,
    prox_norm1,
    prox_norm2,
    prox_norm_nuc,
    prox_sum_exp,
    prox_sum_largest,
    prox_inv_pos,
    prox_huber,
    prox_max_entries,
    prox_norm12,
    prox_neg_log_det,
    prox_negative_log,
    prox_negative_entropy,
    prox_hinge,
    prox_norm_l1_asymmetric,
    prox_deadzone,
    prox_norm_l1_asymmetric_single_max,
    prox_matrix_frac,
    prox_max_elementwise,
    prox_linear_equality,
    prox_cone,

    # General rewrites
    prox_negate,
    prox_scalar_multiply,
    prox_add,

    # Epigraph rules
    prox_epigraph_atomic,
    prox_linear_epigraph,
    prox_epigraph,
]

def transform_expr(expr):
    for prox_rule in PROX_RULES:
        prox_exprs = list(prox_rule(expr))
        if prox_exprs:
            return prox_exprs

    raise CanonicalizeError("No prox rule", expr)

def transform(input):
    prox_exprs = []
    for constr in input.constraint:
        prox_exprs += list(transform_expr(constr))
    prox_exprs += list(transform_expr(input.objective))
    return Problem(objective=add(*prox_exprs))
