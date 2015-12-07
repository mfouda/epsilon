"""Operations on LinearMaps."""

import numpy as np
import scipy.sparse as sp

from epopt import constant
from epopt.expression_util import *
from epopt.proto.epsilon.expression_pb2 import LinearMap

# Atomic linear maps
def kronecker_product(A, B):
    if A.m*A.n == 1:
        return B
    if B.m*B.n == 1:
        return A

    if (A.linear_map_type == LinearMap.SCALAR and
        B.linear_map_type == LinearMap.SCALAR):
        return scalar(A.scalar*B.scalar, A.n*B.n)

    return LinearMap(
        linear_map_type=LinearMap.KRONECKER_PRODUCT,
        m=A.m*B.m,
        n=A.n*B.n,
        arg=[A, B])

def kronecker_product_single_arg(A, Bm, Bn):
    return LinearMap(
        linear_map_type=LinearMap.KRONECKER_PRODUCT,
        m=A.m*Bm,
        n=A*n*Bn,
        arg=[A])

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

# NOTE(mwytock): Represent the following functions as sparse matrices. This is
# not very efficient, but we expect these to be relatively rare so the sparse
# matrix form should be fine.
def diag_mat(n):
    rows = np.arange(n)
    cols = np.arange(n)*(n+1)
    A = sp.coo_matrix((np.ones(n), (rows, cols)), shape=(n, n*n))
    return sparse_matrix(constant.store(A))

def diag_vec(n):
    rows = np.arange(n)*(n+1)
    cols = np.arange(n)
    A = sp.coo_matrix((np.ones(n), (rows, cols)), shape=(n*n, n))
    return sparse_matrix(constant.store(A))

def trace(n):
    rows = np.zeros(n)
    cols = np.arange(n)*(n+1)
    A = sp.coo_matrix((np.ones(n), (rows, cols)), shape=(1, n*n))
    return sparse_matrix(constant.store(A))

def upper_tri(n):
    m = n*(n-1)/2
    rows = np.arange(m)
    cols = np.array([j*n + i for i in xrange(n) for j in xrange(i+1,n)])
    A = sp.coo_matrix((np.ones(m), (rows, cols)), shape=(m, n*n))
    return sparse_matrix(constant.store(A))
