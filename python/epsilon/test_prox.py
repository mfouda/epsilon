
import unittest
import logging

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from epsilon import cvxpy_expr
from epsilon import prox
from epsilon.prox_pb2 import ProxFunction
from nose.tools import assert_items_equal

def convert(prob):
    prob_proto, data_proto = cvxpy_expr.convert_problem(prob)
    return prox.convert_problem(prob_proto)

def test_lasso():
    m = 10
    n = 5
    lam = 0.1

    A = np.random.randn(m,n)
    b = np.random.randn(m,1)
    x = cp.Variable(n, 1)

    prob = convert(cp.Problem(cp.Minimize(
        cp.sum_squares(A*x - b) + lam*cp.norm(x,1))))

    assert_items_equal(
        (f.function for f in prob.prox_function),
        (ProxFunction.SUM_SQUARES, ProxFunction.NORM_1))

def test_sparse_inverse_covariance():
    n = 10
    lam = 0.1

    S = np.random.randn(n,n)
    W = np.ones((n,n)) - np.eye(n)
    Theta = cp.Variable(n,n)

    prob = convert(cp.Problem(cp.Minimize(
        -cp.log_det(Theta) +
        cp.sum_entries(cp.mul_elemwise(S,Theta)) +
        lam*cp.norm1(cp.mul_elemwise(W,Theta)))))

    assert_items_equal(
        (f.function for f in prob.prox_function),
        (ProxFunction.NEGATIVE_LOG_DET, ProxFunction.NORM_1))

def test_total_variation_1d():
    n = 10
    lam = 1

    b = np.random.randn(n,1)
    x = cp.Variable(n, 1)

    prob = convert(cp.Problem(cp.Minimize(
        cp.sum_squares(x - b) + lam*cp.tv(x))))

    funcs = [f.function for f in prob.prox_function]
    assert_items_equal(
        funcs, (ProxFunction.SUM_SQUARES, ProxFunction.NORM_1))

def test_total_variation_2d():
    n = 10
    lam = 1

    B = np.random.randn(n,n)
    X = cp.Variable(n,n)

    prob = convert(cp.Problem(cp.Minimize(
        cp.sum_squares(X - B) + lam*cp.tv(X))))

    funcs = [f.function for f in prob.prox_function]
    assert_items_equal(
        funcs, (ProxFunction.SUM_SQUARES, ProxFunction.NORM_1_2))
