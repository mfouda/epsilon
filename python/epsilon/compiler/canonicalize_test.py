
import logging
import cvxpy as cp

from epsilon import cvxpy_expr
from epsilon import expression_str
from epsilon.compiler import attributes
from epsilon.compiler import canonicalize
from epsilon.expression_pb2 import Expression
from epsilon.problems import problems_test

ATOMS = [
    cp.norm1,
    cp.norm2,
    cp.sum_squares,
]

def transform(cvxpy_problem):
    input = attributes.transform(
        cvxpy_expr.convert_problem(cvxpy_problem)[0])
    logging.debug("Input:\n%s", expression_str.problem_str(input))
    return canonicalize.transform(input)

def transform_problem(problem_instance):
    output = transform(problem_instance.create())
    logging.debug("Output:\n%s", expression_str.problem_str(output))

def transform_atom(f):
    n = 10
    problem = transform(cp.Problem(cp.Minimize(f(cp.Variable(n)))))

    assert problem.objective.expression_type == Expression.ADD
    assert len(problem.objective.arg) == 1
    assert len(problem.constraint) == 0

def test_atoms():
    for atom in ATOMS:
        yield transform_atom, atom

def test_problems():
    for problem in problems_test.get_problems():
        yield transform_problem, problem
