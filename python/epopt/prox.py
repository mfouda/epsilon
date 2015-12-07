
import logging
import numpy

from epopt import _solve
from epopt import constant
from epopt import cvxpy_expr
from epopt import cvxpy_solver
from epopt import tree_format
from epopt.compiler import compiler
from epopt.compiler import validate
from epopt.error import ProblemError
from epopt.proto.epsilon.expression_pb2 import Expression

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
