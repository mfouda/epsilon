"""CVXPY-like interfaces for solver."""

import logging
import numpy

from cvxpy.settings import OPTIMAL, OPTIMAL_INACCURATE, SOLVER_ERROR

from epsilon import _solve
from epsilon import cvxpy_expr
from epsilon import constant
from epsilon import solver_params_pb2
from epsilon import solver_pb2
from epsilon import util
from epsilon.compiler import compiler
from epsilon.solver_pb2 import SolverStatus

EPSILON = "epsilon"

class SolverError(Exception):
    pass

def set_solution(prob, values):
    for var in prob.variables():
        var_id = cvxpy_expr.variable_id(var)
        assert var_id in values
        x = numpy.fromstring(values[var_id], dtype=numpy.double)
        var.value = x.reshape(var.size[1], var.size[0]).transpose()

def cvxpy_status(solver_status):
    if solver_status.state == SolverStatus.OPTIMAL:
        return OPTIMAL
    elif solver_status.state == SolverStatus.MAX_ITERATIONS_REACHED:
        return OPTIMAL_INACCURATE
    return SOLVER_ERROR

def solve(prob, **kwargs):
    """Solve optimziation problem."""

    if not prob.variables():
        return OPTIMAL

    t0 = util.cpu_time()
    prob_proto = cvxpy_expr.convert_problem(prob)
    prob_proto = compiler.compile_problem(prob_proto)
    t1 = util.cpu_time()
    logging.info("Epsilon compile: %f seconds", t1-t0)

    params = solver_params_pb2.SolverParams(**kwargs)
    if len(prob_proto.objective.arg) == 1 and not prob_proto.constraint:
        # TODO(mwytock): Should probably parameterize the proximal operators so
        # they can take A=0 instead of just using a large lambda here
        lam = 1e12
        values = _solve.eval_prox(
            prob_proto.objective.arg[0].SerializeToString(),
            lam,
            constant.global_data_map,
            {})
        status = OPTIMAL
    else:
        status_str, values = _solve.prox_admm_solve(
            prob_proto.SerializeToString(),
            params.SerializeToString(),
            constant.global_data_map)
        status = cvxpy_status(SolverStatus.FromString(status_str))
    t2 = util.cpu_time()
    logging.info("Epsilon solve: %f seconds", t2-t1)

    set_solution(prob, values)
    return status

def validate_solver(constraints):
    return True

def register_epsilon():
    cvxpy.Problem.register_solve(EPSILON, solve)
