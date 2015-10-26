"""Convert CVXPY expressions into Expression trees."""

import cvxpy
import numpy

from cvxpy import utilities as u
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.mul_elemwise import mul_elemwise
from cvxpy.atoms.affine.sum_entries import sum_entries
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.elementwise.entr import entr
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.atoms.elementwise.huber import huber
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.logistic import logistic
from cvxpy.atoms.elementwise.max_elemwise import max_elemwise
from cvxpy.atoms.elementwise.norm2_elemwise import norm2_elemwise
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.kl_div import kl_div
from cvxpy.atoms.lambda_max import lambda_max
from cvxpy.atoms.log_det import log_det
from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.atoms.matrix_frac import matrix_frac
from cvxpy.atoms.max_entries import max_entries
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.atoms.pnorm import pnorm
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.atoms.sum_largest import sum_largest
from cvxpy.constraints.eq_constraint import EqConstraint
from cvxpy.constraints.leq_constraint import LeqConstraint
from cvxpy.constraints.psd_constraint import PSDConstraint
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variables.variable import Variable
from cvxpy.problems import objective

from epsilon import constant
from epsilon import expression
from epsilon.expression_pb2 import Expression, Size, Problem, Sign, Curvature

def index_value(index, size):
    if index < 0:
        return size + index
    return index

def variable_id(expr):
    return "cvxpy:" + str(expr.id)

def convert_variable(expr):
    m, n = expr.size
    return expression.variable(m, n, variable_id(expr))

def convert_constant(expr):
    m, n = expr.size
    if isinstance(expr.value, (int, long, float)):
        return expression.constant(m, n, scalar=expr.value)
    return expression.constant(m, n, constant=constant.store(expr.value))


def convert_generic(expression_type, expr):
    return Expression(
        expression_type=expression_type,
        size=Size(dim=expr.size),
        curvature=Curvature(
            curvature_type=Curvature.Type.Value(expr.curvature)),
        sign=Sign(
            sign_type=Sign.Type.Value(expr.sign)),
        arg=(convert_expression(arg) for arg in expr.args))

def convert_binary(f, expr):
    return f(*[convert_expression(arg) for arg in expr.args])

def convert_unary(f, expr):
    assert len(expr.args) == 1
    return f(convert_expression(expr.args[0]))

def convert_index(expr):
    starts = []
    stops = []
    assert len(expr.key) == 2
    for i, key in enumerate(expr.key):
        size = expr.args[0].size[i]
        starts.append(index_value(key.start, size) if key.start else 0)
        stops.append(index_value(key.stop, size) if key.stop else size)

    assert len(expr.args) == 1
    return expression.index(convert_expression(expr.args[0]),
                            starts[0], stops[0],
                            starts[1], stops[1])

def convert_huber(expr):
    proto = convert_generic(Expression.HUBER, expr)
    proto.M = expr.M.value
    return proto

def convert_pnorm(expr):
    proto = convert_generic(Expression.NORM_P, expr)
    proto.p = expr.p
    return proto

def convert_power(expr):
    proto = convert_generic(Expression.POWER, expr)
    try:
        proto.p = expr.p
    except TypeError: # FIXME expr.p has type Fraciton on inv_pos
        proto.p = -1

    return proto

def convert_sum_largest(expr):
    proto = convert_generic(Expression.SUM_LARGEST, expr)
    proto.k = expr.k
    return proto

EXPRESSION_TYPES = (
    (AddExpression, lambda e: convert_binary(expression.add, e)),
    (Constant, convert_constant),
    (MulExpression, lambda e: convert_binary(expression.multiply, e)),
    (NegExpression, lambda e: convert_unary(expression.negate, e)),
    (Variable, convert_variable),
    (exp, lambda e: convert_generic(Expression.EXP, e)),
    (entr, lambda e: convert_generic(Expression.ENTR, e)),
    (hstack, lambda e: convert_generic(Expression.HSTACK, e)),
    (huber, convert_huber),
    (index, convert_index),
    (kl_div, lambda e: convert_generic(Expression.KL_DIV, e)),
    (lambda_max, lambda e: convert_generic(Expression.LAMBDA_MAX, e)),
    (log, lambda e: convert_generic(Expression.LOG, e)),
    (log_det, lambda e: convert_generic(Expression.LOG_DET, e)),
    (log_sum_exp, lambda e: convert_generic(Expression.LOG_SUM_EXP, e)),
    (logistic, lambda e: convert_generic(Expression.LOGISTIC, e)),
    (max_elemwise, lambda e: convert_generic(Expression.MAX_ELEMENTWISE, e)),
    (max_entries, lambda e: convert_generic(Expression.MAX_ENTRIES, e)),
    (matrix_frac, lambda e: convert_generic(Expression.MATRIX_FRAC, e)),
    (mul_elemwise, lambda e: convert_binary(expression.multiply_elemwise, e)),
    (norm2_elemwise, lambda e: convert_generic(Expression.NORM_2_ELEMENTWISE, e)),
    (normNuc, lambda e: convert_generic(Expression.NORM_NUC, e)),
    (pnorm, convert_pnorm),
    (power, convert_power),
    (quad_over_lin, lambda e: convert_generic(Expression.QUAD_OVER_LIN, e)),
    (sum_entries, lambda e: convert_generic(Expression.SUM, e)),
    (sum_largest, lambda e: convert_sum_largest(e)),
    (trace, lambda e: convert_generic(Expression.TRACE, e)),
    (transpose, lambda e: convert_unary(expression.transpose, e)),
    (vstack, lambda e: convert_generic(Expression.VSTACK, e)),
)

# Sanity check to make sure the CVXPY atoms are all classes. This can change
# periodically due to implementation details of CVXPY.
import inspect
for expr_cls, expr_type in EXPRESSION_TYPES:
    assert inspect.isclass(expr_cls), expr_cls

def convert_expression(expr):
    for expr_cls, convert in EXPRESSION_TYPES:
        if isinstance(expr, expr_cls):
            return convert(expr)
    raise RuntimeError("Unknown type: %s" % type(expr))

def convert_constraint(constraint):
    if isinstance(constraint, EqConstraint):
        return expression.equality_constraint(
            convert_expression(constraint.args[0]),
            convert_expression(constraint.args[1]))
    elif isinstance(constraint, PSDConstraint):
        return expression.psd_constraint(
            convert_expression(constraint.args[0]),
            convert_expression(constraint.args[1]))
    elif isinstance(constraint, LeqConstraint):
        return expression.leq_constraint(
            convert_expression(constraint.args[0]),
            convert_expression(constraint.args[1]))

    raise RuntimeError("Unknown constraint: %s" % type(constraint))

def convert_problem(problem):
    # NOTE(mwytock): Maximize inherits from Minimize so this clause first!
    if isinstance(problem.objective, objective.Maximize):
        obj_expr = -problem.objective.args[0]
    elif isinstance(problem.objective, objective.Minimize):
        obj_expr = problem.objective.args[0]
    else:
        raise RuntimeError("Unknown objective: %s" % type(problem.objective))

    return Problem(
        objective=convert_expression(obj_expr),
        constraint=[convert_constraint(c) for c in problem.constraints])
