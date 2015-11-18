
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

def eval_prox(prox_function_type, prob, v_map, lam=1):
    """Evaluate a single proximal operator."""

    problem = compiler.compile_problem(cvxpy_expr.convert_problem(prob))
    validate.check_sum_of_prox(problem)

    non_const = []
    for f_expr in problem.objective.arg:
        if f_expr.curvature.curvature_type != Curvature.CONSTANT:
            non_const.append(f_expr)

    # Get the first non constant objective term
    if len(non_const) != 1:
        raise ProblemError("prox does not have single f", problem)

    if (non_const[0].expression_type != Expression.PROX_FUNCTION or
        non_const[0].prox_function.prox_function_type != prox_function_type):
        raise ProblemError("prox did not compile to right type", problem)

    if problem.constraint:
        raise ProblemError("prox has constraints", problem)

    v_bytes_map = {cvxpy_expr.variable_id(var): val.tobytes(order="F") for
                   var, val in v_map.iteritems()}
    values = _solve.eval_prox(
        non_const[0].SerializeToString(),
        lam,
        constant.global_data_map,
        v_bytes_map)

    cvxpy_solver.set_solution(prob, values)
