"""Manipulate cvxpy expressions."""

import cvxpy
import numpy

from cvxpy import utilities as u
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.mul_elemwise import mul_elemwise
from cvxpy.atoms.affine.sum_entries import sum_entries
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.elementwise.norm2_elemwise import norm2_elemwise
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.elementwise.max_elemwise import max_elemwise
from cvxpy.atoms.log_det import log_det
from cvxpy.atoms.pnorm import pnorm
from cvxpy.constraints.eq_constraint import EqConstraint
from cvxpy.constraints.leq_constraint import LeqConstraint
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variables.variable import Variable

from epsilon import data
from epsilon import expression_pb2
from epsilon import problem_pb2
from epsilon.expression_pb2 import Expression as E

EXPRESSION_TYPES = (
    (AddExpression, E.ADD),
    (Constant, E.CONSTANT),
    (MulExpression, E.MULTIPLY),
    (NegExpression, E.NEGATE),
    (Variable, E.VARIABLE),
    (index, E.INDEX),
    (log_det, E.LOG_DET),
    (max_elemwise, E.MAX),
    (mul_elemwise, E.MULTIPLY_ELEMENTWISE),
    (norm2_elemwise, E.NORM_2_ELEMENTWISE),
    (pnorm, E.P_NORM),
    (power, E.POWER),
    (sum_entries, E.SUM),
    (trace, E.TRACE),
    (transpose, E.TRANSPOSE),
)

import inspect
for expr_cls, expr_type in EXPRESSION_TYPES:
    assert inspect.isclass(expr_cls), expr_cls


class DistributedConstant(Constant):
    def __init__(self, value):
        self._value = value
        shape = u.Shape(value.m, value.n)
        self._dcp_attr = u.DCPAttr(u.Sign.UNKNOWN, u.Curvature.CONSTANT, shape)

def variable_id(var):
    return "cvxpy:" + str(var.id)

def convert_constant(value, proto, data_map):
    if isinstance(value, (int, long, float)):
        proto.scalar = value
        return

    assert isinstance(value, numpy.ndarray)
    prefix = "/mem/data/" + str(abs(hash(value.tostring())))
    data_map[data.metadata_file(prefix)] = (
        data.dense_matrix_metadata(value).SerializeToString())
    data_map[data.value_file(prefix)] = value.tobytes(order="Fortran")
    proto.data_location = prefix

def convert_expression(expr, proto, data_map):
    """Convert cxvpy expression to protobuf form."""
    proto.size.dim.extend(expr.size)
    proto.curvature.curvature_type = (
        expression_pb2.Curvature.Type.Value(expr.curvature))
    proto.sign.sign_type = expression_pb2.Sign.Type.Value(expr.sign)

    for arg in getattr(expr, "args", []):
        convert_expression(arg, proto.arg.add(), data_map)

    if isinstance(expr, Constant):
        convert_constant(expr.value, proto.constant, data_map)
    elif isinstance(expr, Variable):
        proto.variable.variable_id = variable_id(expr)
    elif isinstance(expr, index):
        for i, key in enumerate(expr.key):
            key_proto = proto.key.add()

            if key.start:
                key_proto.start = key.start

            if key.stop:
                key_proto.stop = key.stop
            else:
                key_proto.stop = expr.size[i]

            if key.step:
                key_proto.step = key.step
            else:
                key_proto.step = 1
    elif isinstance(expr, (power, pnorm)):
        proto.p = expr.p

    for expr_cls, expr_type in EXPRESSION_TYPES:
        if isinstance(expr, expr_cls):
            proto.expression_type = expr_type
            break
    else:
        raise RuntimeError("Unknown type: %s" % type(expr))

def convert_objective(objective, proto, data_map):
    # TODO(mwytock): Handle Maximize()
    convert_expression(objective.args[0], proto, data_map)

def convert_constraint(constraint, proto, data_map):
    proto.constraint_id = "constraint:%d" % constraint.constr_id
    if isinstance(constraint, EqConstraint):
        proto.cone = expression_pb2.ZERO
        convert_expression(constraint._expr, proto.arg.add(), data_map)
    elif isinstance(constraint, LeqConstraint):
        proto.cone = expression_pb2.NON_NEGATIVE
        convert_expression(constraint._expr, proto.arg.add(), data_map)
    else:
        raise RuntimeError("Unknown constraint: %s" % type(constraint))

def convert_problem(problem):
    proto = problem_pb2.Problem()
    data_map = {}
    convert_objective(problem.objective, proto.objective, data_map)
    for constraint in problem.constraints:
        convert_constraint(constraint, proto.constraint.add(), data_map)
    return proto, data_map

def convert_matrix_problem(problem):
    prob_data = problem.get_problem_data(solver=cvxpy.SCS)

    proto = data_pb2.MatrixProblem()
    data.fill_sparse_matrix(prob_data["A"], proto.A)
    data.fill_vector(prob_data["b"], proto.b)
    data.fill_vector(prob_data["c"], proto.c)

    dims = prob_data["dims"]
    proto.cone_constraints.equality = dims["f"]
    proto.cone_constraints.inequality = dims["l"]
    proto.cone_constraints.second_order_cone.extend(dims["q"])
    proto.cone_constraints.semi_definite.extend(dims["s"])

    return proto