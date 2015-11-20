"""CVXPY-like interfaces for solver."""

import logging
import numpy

from cvxpy.settings import OPTIMAL

from epsilon import _solve
from epsilon import cvxpy_expr
from epsilon import constant
from epsilon import solver_params_pb2
from epsilon import solver_pb2
from epsilon.compiler import compiler

EPSILON = "epsilon"

class SolverError(Exception):
    pass

def set_solution(prob, values):
    for var in prob.variables():
        var_id = cvxpy_expr.variable_id(var)
        assert var_id in values
        x = numpy.fromstring(values[var_id], dtype=numpy.double)
        var.value = x.reshape(var.size[1], var.size[0]).transpose()

def solve(prob, rel_tol=1e-2, abs_tol=1e-4):
    """Solve optimziation problem."""

    if not prob.variables():
        return OPTIMAL, prob.objective.value

    prob_proto = cvxpy_expr.convert_problem(prob)
    prob_proto = compiler.compile_problem(prob_proto)

    params = solver_params_pb2.SolverParams(
        rel_tol=rel_tol, abs_tol=abs_tol)
    if len(prob_proto.objective.arg) == 1:
        # TODO(mwytock): Should probably parameterize the proximal operators so
        # they can take A=0 instead of just using a large lambda here
        lam = 1e12
        values = _solve.eval_prox(
            prob_proto.objective.arg[0].SerializeToString(),
            lam,
            constant.global_data_map,
            {})
    else:
        status_str, values = _solve.prox_admm_solve(
            prob_proto.SerializeToString(),
            params.SerializeToString(),
            constant.global_data_map)

    # TODO(mwytock): Handle not optimal solutions
    set_solution(prob, values)
    return OPTIMAL, prob.objective.value

def validate_solver(constraints):
    return True

# def prox(cvxpy_prob, v_map, lam=1):
#     """Evaluate a single proximal operator."""

#     problem = cvxpy_expr.convert_problem(cvxpy_prob)
#     logging.debug("Input:\n%s", tree_format.format_problem(problem))
#     problem = canonicalize_linear.transform_problem(
#         canonicalize.transform(problem))
#     logging.debug("Canonical:\n%s", tree_format.format_problem(problem))
#     validate.check_sum_of_prox(problem)

#     non_const = []
#     for f_expr in problem.objective.arg:
#         if f_expr.curvature.curvature_type != Curvature.CONSTANT:
#             non_const.append(f_expr)

#     # Get the first non constant objective term
#     if len(non_const) != 1:
#         raise ProblemError("prox does not have single f", problem)

#     if problem.constraint:
#         raise ProblemError("prox has constraints", problem)

#     v_bytes_map = {cvxpy_expr.variable_id(var): val.tobytes(order="F") for
#                    var, val in v_map.iteritems()}
#     values = _solve.prox(
#         non_const[0].SerializeToString(),
#         lam,
#         constant.global_data_map,
#         v_bytes_map)

#     for var in cvxpy_prob.variables():
#         var_id = cvxpy_expr.variable_id(var)
#         assert var_id in values
#         x = numpy.fromstring(values[var_id], dtype=numpy.double)
#         var.value = x.reshape(var.size[1], var.size[0]).transpose()

def register_epsilon():
    cvxpy.Problem.register_solve(EPSILON, solve)
