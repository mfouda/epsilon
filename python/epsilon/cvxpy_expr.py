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
from cvxpy.atoms.lambda_max import lambda_max
from cvxpy.atoms.log_det import log_det
from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.atoms.max_entries import max_entries
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.atoms.pnorm import pnorm
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.constraints.eq_constraint import EqConstraint
from cvxpy.constraints.leq_constraint import LeqConstraint
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variables.variable import Variable

from epsilon import data
from epsilon import expression
from epsilon.expression_pb2 import Expression, Size, Problem, Sign, Curvature

def index_value(index, size):
    if index < 0:
        return size + index
    return index

def variable_id(expr):
    return "cvxpy:" + str(expr.id)

def value_location(value):
    return "/mem/data/" + str(abs(hash(value.tostring())))

def convert_variable(expr):
    m, n = expr.size
    return expression.variable(m, n, variable_id(expr))

def convert_constant(expr):
    m, n = expr.size
    if isinstance(expr.value, (int, long, float)):
        return expression.constant(m, n, scalar=expr.value)
    assert isinstance(expr.value, numpy.ndarray)
    return expression.constant(m, n, data_location=value_location(expr.value))

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
    (lambda_max, lambda e: convert_generic(Expression.LAMBDA_MAX, e)),
    (log, lambda e: convert_generic(Expression.LOG, e)),
    (log_det, lambda e: convert_generic(Expression.LOG_DET, e)),
    (log_sum_exp, lambda e: convert_generic(Expression.LOG_SUM_EXP, e)),
    (logistic, lambda e: convert_generic(Expression.LOGISTIC, e)),
    (max_elemwise, lambda e: convert_generic(Expression.MAX_ELEMENTWISE, e)),
    (max_entries, lambda e: convert_generic(Expression.MAX_ENTRIES, e)),
    (mul_elemwise, lambda e: convert_binary(expression.multiply_elemwise, e)),
    (norm2_elemwise, lambda e: convert_generic(Expression.NORM_2_ELEMENTWISE, e)),
    (normNuc, lambda e: convert_generic(Expression.NORM_NUC, e)),
    (pnorm, convert_pnorm),
    (power, convert_power),
    (quad_over_lin, lambda e: convert_generic(Expression.QUAD_OVER_LIN, e)),
    (sum_entries, lambda e: convert_generic(Expression.SUM, e)),
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
    if isinstance(constraint, LeqConstraint):
        return expression.leq_constraint(
            convert_expression(constraint.args[0]),
            convert_expression(constraint.args[1]))

    raise RuntimeError("Unknown constraint: %s" % type(constraint))

def add_expression_data(expr, data_map):
    if isinstance(expr, Constant) and isinstance(expr.value, numpy.ndarray):
        prefix = value_location(expr.value)
        data_map[data.metadata_file(prefix)] = (
            data.dense_matrix_metadata(expr.value).SerializeToString())
        data_map[data.value_file(prefix)] = expr.value.tobytes(order="F")

    for arg in getattr(expr, "args", []):
        add_expression_data(arg, data_map)

def extract_data(problem):
    data_map = {}
    for arg in problem.objective.args:
        add_expression_data(arg, data_map)
    for constraint in problem.constraints:
        for arg in constraint.args:
            add_expression_data(arg, data_map)
    return data_map

# TODO(mwytock): Assumes minimize, handle maximize()
def convert_problem(problem):
    proto = Problem(
        objective=convert_expression(problem.objective.args[0]),
        constraint=[convert_constraint(c) for c in problem.constraints])
    data_map = extract_data(problem)
    return proto, data_map
