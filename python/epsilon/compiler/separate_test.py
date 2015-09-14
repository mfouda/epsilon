
import logging
import cvxpy as cp

from epsilon import cvxpy_expr
from epsilon import expression_str
from epsilon.compiler import attributes
from epsilon.compiler import canonicalize
from epsilon.compiler import separate
from epsilon.expression_pb2 import Expression
from epsilon.problems import problems_test

def transform(cvxpy_problem):
    input = canonicalize.transform(
        attributes.transform(
            cvxpy_expr.convert_problem(cvxpy_problem)[0]))

    logging.debug("Input:\n%s", expression_str.problem_str(input))
    return separate.transform(input)

def transform_problem(problem_instance):
    output = transform(problem_instance.create())
    logging.debug("Output:\n%s", expression_str.problem_str(output))

def test_problems():
    for problem in problems_test.get_problems():
        yield transform_problem, problem
