
import logging
import numpy

from epsilon import _solve
from epsilon import cvxpy_expr
from epsilon import solver_params_pb2
from epsilon import solver_pb2
from epsilon.compiler import attributes
from epsilon.compiler import canonicalize
from epsilon.compiler import compiler
from epsilon.compiler import validate
from epsilon.error import ProblemError
from epsilon.expression_pb2 import Curvature

def solve(prob, params=solver_params_pb2.SolverParams()):
    """Solve optimziation problem."""

    prob_proto, data_map = cvxpy_expr.convert_problem(prob)
    prob_proto = compiler.compile(prob_proto)

    status_str, values = _solve.prox_admm_solve(
        prob_proto.SerializeToString(),
        params.SerializeToString(),
        data_map)

    for var in prob.variables():
        var_id = cvxpy_expr.variable_id(var)
        assert var_id in values
        x = numpy.fromstring(values[var_id], dtype=numpy.double)
        var.value = x.reshape(var.size[1], var.size[0]).transpose()

    return solver_pb2.SolverStatus.FromString(status_str)

def prox(cvxpy_prob, v, lam=1):
    """Evaluate a single proximal operator."""

    problem, data_map = cvxpy_expr.convert_problem(cvxpy_prob)
    problem = canonicalize.transform(attributes.transform(problem))
    validate.check_sum_of_prox(problem)

    non_const = []
    for f_expr in problem.objective.arg:
        if f_expr.curvature.curvature_type != Curvature.CONSTANT:
            non_const.append(f_expr)

    # Get the first non constant objective term
    if len(non_const) != 1:
        raise ProblemError("prox does not have single f", problem)

    if problem.constraint:
        raise ProblemError("prox has constraints", problem)

    x_bytes = _solve.prox(
        non_const[0].SerializeToString(), data_map, v.tobytes(order="F"), lam)
    return numpy.fromstring(x_bytes, dtype=numpy.double)
