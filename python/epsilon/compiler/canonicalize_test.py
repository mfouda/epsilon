
import logging
import cvxpy as cp
import numpy as np

from epsilon import cvxpy_expr
from epsilon import expression_str
from epsilon.compiler import canonicalize
from epsilon.expression_pb2 import Expression

def transform(cvxpy_problem):
    input = cvxpy_expr.convert_problem(cvxpy_problem)[0]
    logging.debug("Input:\n%s", expression_str.problem_str(input))
    return canonicalize.transform(input)

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
