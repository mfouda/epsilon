
import logging
import cvxpy as cp

from epsilon import cvxpy_expr
from epsilon.compiler import canonicalize
from epsilon.compiler import attributes
from epsilon.problems import problems_test
from epsilon.expression_pb2 import Expression

ATOMS = [
    cp.norm1,
    cp.norm2,
    cp.sum_squares,
]

def transform(cvxpy_problem):
    input = attributes.transform(
        cvxpy_expr.convert_problem(cvxpy_problem)[0])
    logging.debug("Input problem:\n%s", input)

    return canonicalize.transform(input)

def transform_problem(problem_instance):
    transform(problem_instance.create())

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
