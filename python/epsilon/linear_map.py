"""Operations on LinearMaps."""

import numpy as np
import scipy.sparse as sp

from epsilon import data
from epsilon.expression_pb2 import LinearMap
from epsilon.expression_util import *

# Atomic linear maps
def kronecker_product(A, B):
    if A.m*A.n == 1:
        return B
    if B*m*B*n == 1:
        return A

    return LinearMap(
        linear_map_type=LinearMap.KRONECKER_PRODUCT,
        m=A.m*B.m,
        n=A.n*B.n,
        arg=[A, B])

def dense_matrix(constant_expr):
    return LinearMap(
        linear_map_type=LinearMap.DENSE_MATRIX,
        m=dim(constant_expr, 0),
        n=dim(constant_expr, 1),
        constant=constant_expr.constant)

def sparse_matrix(constant_expr):
    return LinearMap(
        linear_map_type=LinearMap.SPARSE_MATRIX,
        m=dim(constant_expr, 0),
        n=dim(constant_expr, 1),
        constant=constant_expr.constant)

def diagonal_matrix(constant_expr):
    return LinearMap(
        linear_map_type=LinearMap.DIAGONAL_MATRIX,
        m=dim(constant_expr),
        n=dim(constant_expr),
        constant=constant_expr.constant)

def scalar(alpha, n):
    return LinearMap(
        linear_map_type=LinearMap.SCALAR,
        m=n,
        n=n,
        scalar=alpha)

# Operations on linear maps
def transpose(A):
    return LinearMap(
        linear_map_type=LinearMap.TRANSPOSE,
        m=A.n,
        n=A.m,
        A=A)

# Implementation of various linear maps in terms of atoms
def identity(n):
    return scalar(1, n)

def index(slice, n):
    m = slice.stop - slice.start
    A = sp.coo_matrix(
        np.ones(n),
        (np.arange(m), np.arange(slice.start, m, slice.step)))
    return sparse_matrix(data.store_constant(A))

def one_hot(i, n):
    """[0, ... 0, 1, 0, ...]."""
    a = sp.coo_matrix(([1], ([0], [i])), shape=(1, n))
    return sparse_matrix(data.store_constant(a))

def sum(n):
    """All ones vector"""
    return dense_matrix(constants.store(np.ones(n)))

def negate(n):
    return scalar(-1,n)

def left_matrix_product(A, n):
    return kronecker_product(identity(n), A)

def right_matrix_product(B, m):
    return kronecker_product(transpose(B), identity(m))
