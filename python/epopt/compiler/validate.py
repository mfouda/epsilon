

from epopt.expression_pb2 import Expression, Problem
from epopt import error

class ValidateError(error.ProblemError):
    pass


def check_sum_of_prox(problem):
    if problem.objective.expression_type != Expression.ADD:
        raise ValidateError("Objective is not ADD", problem)
