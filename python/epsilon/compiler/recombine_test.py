
import logging
import cvxpy as cp

from epsilon import cvxpy_expr
from epsilon import expression_str
from epsilon.compiler import attributes
from epsilon.compiler import canonicalize
from epsilon.compiler import recombine
from epsilon.expression import *
from epsilon.expression_pb2 import *
from epsilon.problems import problems_test

def transform_problem(problem_instance):
    input = canonicalize.transform(
        attributes.transform(
            cvxpy_expr.convert_problem(problem_instance.create())[0]))
    logging.debug("Input:\n%s", expression_str.problem_str(input))

    output = recombine.transform(problem_instance.create())
    logging.debug("Output:\n%s", expression_str.problem_str(output))

# def test_problems():
#     for problem in problems_test.get_problems():
#         yield transform_problem, problem


x = variable(1, 1, "x")

def test_merge_single_affine():
    prob = recombine.merge_affine(Problem(objective=add(x)))
    assert prob.objective.expression_type == Expression.ADD
    assert len(prob.objective.arg) == 1
    assert prob.objective.expression_type == Expression.VARIABLE

def test_merge_single_non_affine():
    prob = recombine.merge_affine(Problem(objective=add(power(x, 2))))
    assert prob.objective.expression_type == Expression.ADD
    assert len(prob.objective.arg) == 1
    assert prob.objective.expression_type == Expression.NORM_P

def test_merge_mixed_pair():
    prob = recombine.merge_affine(
        Problem(objective=add(power(x, 2), x)))
    assert prob.objective.expression_type == Expression.ADD
    assert len(prob.objective.arg) == 1
    assert prob.objective.expression_type == Expression.ADD
