"""CVXPY-like interfaces for solver."""

import logging
import numpy

from cvxpy.settings import OPTIMAL, OPTIMAL_INACCURATE, SOLVER_ERROR

from epopt import _solve
from epopt import constant
from epopt import cvxpy_expr
from epopt import util
from epopt.compiler import compiler
from epopt.proto.epsilon import solver_params_pb2
from epopt.proto.epsilon import solver_pb2
from epopt.proto.epsilon.solver_pb2 import SolverStatus

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

def solve(cvxpy_prob, **kwargs):
    """Solve optimziation problem."""

    # Nothing to do in this case
    if not cvxpy_prob.variables():
        return cvxpy_prob.objective.value

    t0 = util.cpu_time()
    problem = cvxpy_expr.convert_problem(cvxpy_prob)
    problem = compiler.compile_problem(problem)
    t1 = util.cpu_time()
    logging.info("Epsilon compile: %f seconds", t1-t0)

    if len(problem.objective.arg) == 1 and not problem.constraint:
        # TODO(mwytock): Should probably parameterize the proximal operators so
        # they can take A=0 instead of just using a large lambda here
        lam = 1e12
        values = _solve.eval_prox(
            problem.objective.arg[0].SerializeToString(),
            lam,
            constant.global_data_map,
            {})
        status = OPTIMAL
    else:
        params = solver_params_pb2.SolverParams(**kwargs)
        status_str, values = _solve.prox_admm_solve(
            problem.SerializeToString(),
            params.SerializeToString(),
            constant.global_data_map)
        status = cvxpy_status(SolverStatus.FromString(status_str))
    t2 = util.cpu_time()
    logging.info("Epsilon solve: %f seconds", t2-t1)

    set_solution(cvxpy_prob, values)
    return cvxpy_prob.objective.value

def validate_solver(constraints):
    return True

def register_epsilon():
    cvxpy.Problem.register_solve(EPSILON, solve)
