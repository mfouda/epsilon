

from epsilon.expression_pb2 import Expression, Problem
from epsilon.compiler import compiler_error

class ValidateError(compiler_error.CompilerError):
    pass


def check_sum_of_prox(problem):
    if problem.objective.expression_type != Expression.ADD:
        raise ValidateError("Objective is not ADD", problem)

