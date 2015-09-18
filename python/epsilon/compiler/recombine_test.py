
import logging
import cvxpy as cp

from epsilon import cvxpy_expr
from epsilon import expression_str
from epsilon.compiler import attributes
from epsilon.compiler import canonicalize
from epsilon.compiler import recombine
from epsilon.expression import *
from epsilon.expression_pb2 import *

x = variable(1, 1, "x")
def test_merge_single_affine():
    prob = recombine.merge_affine(Problem(objective=add(x)))
    assert prob.objective.expression_type == Expression.ADD
    assert len(prob.objective.arg) == 1
    assert prob.objective.arg[0].expression_type == Expression.VARIABLE

def test_merge_single_non_affine():
    prob = recombine.merge_affine(Problem(objective=add(power(x, 2))))
    assert prob.objective.expression_type == Expression.ADD
    assert len(prob.objective.arg) == 1
    assert prob.objective.arg[0].expression_type == Expression.POWER

def test_merge_mixed_pair():
    prob = recombine.merge_affine(
        Problem(objective=add(power(x, 2), x)))
    assert prob.objective.expression_type == Expression.ADD
    assert len(prob.objective.arg) == 1
    assert prob.objective.expression_type == Expression.ADD
