
import logging
import cvxpy as cp
import numpy as np

from epsilon import cvxpy_expr
from epsilon import expression_str
from epsilon.compiler import attributes
from epsilon.compiler import canonicalize
from epsilon.expression_pb2 import Expression

ATOMS = [
    cp.norm1,
    cp.norm2,
    cp.sum_squares,
]

def transform(cvxpy_problem):
    input = cvxpy_expr.convert_problem(cvxpy_problem)[0]
    logging.debug("Input:\n%s", expression_str.problem_str(input))
    return canonicalize.transform(input)

def transform_atom(f):
    n = 10
    problem = transform(cp.Problem(cp.Minimize(f(cp.Variable(n)))))

    assert problem.objective.expression_type == Expression.ADD
    assert len(problem.objective.arg) == 1
    assert len(problem.constraint) == 0

def test_atoms():
    for atom in ATOMS:
        yield transform_atom, atom

def test_composite_epigraph():
    n = 5
    c = np.arange(n)
    x = cp.Variable(n)
    f = cp.exp(cp.norm2(x) + cp.norm1(x) + c.T*x) + cp.norm2(x)
    problem = transform(cp.Problem(cp.Minimize(f)))

def test_multiply_scalar():
    n = 5
    x = cp.Variable(n)
    f = cp.sum_entries(0.25*cp.max_elemwise(x, 0))
    problem = transform(cp.Problem(cp.Minimize(f)))
