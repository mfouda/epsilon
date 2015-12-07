
import logging
import numpy

from epsilon import _solve
from epsilon import constant
from epsilon import cvxpy_expr
from epsilon import cvxpy_solver
from epsilon import solver_params_pb2
from epsilon import solver_pb2
from epsilon import tree_format
from epsilon.compiler import compiler
from epsilon.compiler import validate
from epsilon.error import ProblemError
from epsilon.expression_pb2 import Curvature, Expression, ProxFunction

def eval_prox(prox_function_type, prob, v_map, lam=1, epigraph=False):
    """Evaluate a single proximal operator."""

    problem = compiler.compile_problem(cvxpy_expr.convert_problem(prob))
    validate.check_sum_of_prox(problem)

    # Get the first non constant objective term
    if len(problem.objective.arg) != 1:
        raise ProblemError("prox does not have single f", problem)

    if problem.constraint:
        raise ProblemError("prox has constraints", problem)

    f_expr = problem.objective.arg[0]
    if (f_expr.expression_type != Expression.PROX_FUNCTION or
        f_expr.prox_function.prox_function_type != prox_function_type or
        f_expr.prox_function.epigraph != epigraph):
        raise ProblemError("prox did not compile to right type", problem)

    v_bytes_map = {cvxpy_expr.variable_id(var):
                   numpy.array(val, dtype=numpy.float64).tobytes(order="F")
                   for var, val in v_map.iteritems()}

    values = _solve.eval_prox(
        f_expr.SerializeToString(),
        lam,
        constant.global_data_map,
        v_bytes_map)

    cvxpy_solver.set_solution(prob, values)
