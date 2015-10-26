"""Operations on LinearMaps."""

import numpy as np
import scipy.sparse as sp

from epsilon import constant
from epsilon.expression_pb2 import LinearMap
from epsilon.expression_util import *

# Atomic linear maps
def kronecker_product(A, B):
    if A.m*A.n == 1:
        return B
    if B.m*B.n == 1:
        return A

    return LinearMap(
        linear_map_type=LinearMap.KRONECKER_PRODUCT,
        m=A.m*B.m,
        n=A.n*B.n,
        arg=[A, B])

def dense_matrix(constant):
    return LinearMap(
        linear_map_type=LinearMap.DENSE_MATRIX,
        m=constant.m,
        n=constant.n,
        constant=constant)

def sparse_matrix(constant):
    return LinearMap(
        linear_map_type=LinearMap.SPARSE_MATRIX,
        m=constant.m,
        n=constant.n,
        constant=constant)

def diagonal_matrix(constant):
    n = constant.m*constant.n
    return LinearMap(
        linear_map_type=LinearMap.DIAGONAL_MATRIX, m=n, n=n, constant=constant)

def scalar(alpha, n):
    return LinearMap(
        linear_map_type=LinearMap.SCALAR,
        m=n,
        n=n,
        scalar=alpha)

# Operations on linear maps
def transpose(A):
    return LinearMap(
        linear_map_type=LinearMap.TRANSPOSE, m=A.n, n=A.m, arg=[A])

# Implementation of various linear maps in terms of atoms
def identity(n):
    return scalar(1, n)

def index(slice, n):
    m = slice.stop - slice.start
    if m == n:
        return identity(n)

    A = sp.coo_matrix(
        (np.ones(m),
         (np.arange(m), np.arange(slice.start, slice.stop, slice.step))),
        shape=(m, n))
    return sparse_matrix(constant.store(A))

def one_hot(i, n):
    return sparse_matrix(
        constant.store(sp.coo_matrix(([1], ([i], [0])), shape=(n,1))))

def sum(n):
    return dense_matrix(constant.store(np.ones((1,n))))

def promote(n):
    return dense_matrix(constant.store(np.ones((n,1))))

def negate(n):
    return scalar(-1,n)

def left_matrix_product(A, n):
    return kronecker_product(identity(n), A)

def right_matrix_product(B, m):
    return kronecker_product(transpose(B), identity(m))
