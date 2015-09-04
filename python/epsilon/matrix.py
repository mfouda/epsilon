

import logging

from distopt import data
from distopt import data_pb2
from distopt import distributed_file

METADATA_FILE = "metadata"
VALUE_FILE = "value"
VALUE_TRANSPOSE_FILE = "value_transpose"

def metadata_file(prefix):
    return prefix + "/" + METADATA_FILE

def value_file(prefix):
    return prefix + "/" + VALUE_FILE

def value_transpose_file(prefix):
    return prefix + "/" + VALUE_TRANSPOSE_FILE

def write_matrix(prefix, A):
    logging.info("Writing metadata")
    with  distributed_file.open(metadata_file(prefix), "w") as f:
        f.write(data.dense_matrix_metadata(A).SerializeToString())

    logging.info("Writing data")
    with  distributed_file.open(value_file(prefix), "w") as f:
        f.write(A.tobytes(order="Fortran"))  # Column order

    logging.info("Writing data transpose")
    with distributed_file.open(value_transpose_file(prefix), "w") as f:
        f.write(A.tobytes(order="C"))        # Row order

class Matrix(object):
    """A distirbuted matrix."""

    def __init__(self, location):
        self.location = location
        with distributed_file.open(metadata_file(location)) as f:
            self.metadata = data_pb2.Data.FromString(f.read())

        self.m = self.metadata.dense_matrix.m
        self.n = self.metadata.dense_matrix.n


class Vector(Matrix):
    def __init__(self, prefix):
        Matrix.__init__(self, prefix)
        assert self.n == 1


# Quick test
if __name__ == "__main__":
    import distopt
    import cvxpy as cp
    import numpy as np

    # NOTE(mwytock): Need to explicitly create constant expressions here to get
    # around cvxpy assumptions about dense/sparse constants
    A_ = distopt.Matrix("/s3/us-west-2/distopt/medium/A")
    b_ = distopt.Vector("/s3/us-west-2/distopt/medium/b")
    A = distopt.cvxpy_expr.DistributedConstant(A_)
    b = distopt.cvxpy_expr.DistributedConstant(b_)

    x = cp.Variable(A_.n)
    y = cp.Variable(A_.m)

    prob = cp.Problem(cp.Minimize(cp.norm(A*x - b)))
    print distopt.cvxpy_expr.convert_problem(prob)
