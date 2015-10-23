
import numpy
from scipy import sparse

from epsilon import data_pb2
from epsilon import expression

METADATA_FILE = "metadata"
VALUE_FILE = "value"

data_map = {}

def metadata_file(prefix):
    return prefix + "/" + METADATA_FILE

def value_file(prefix):
    return prefix + "/" + VALUE_FILE

def fill_vector(value, vector):
    vector.value_bytes = value.tobytes(order="Fortran")

def fill_sparse_matrix(A, sparse_matrix):
    sparse_matrix.m, sparse_matrix.n = A.shape
    sparse_matrix.pr.extend(float(x) for x in A.data)
    sparse_matrix.jc.extend(int(x) for x in A.indptr)
    sparse_matrix.ir.extend(int(x) for x in A.indices)

def dense_matrix_data(value):
    data = data_pb2.Data()
    data.data_type = data_pb2.Data.DENSE_MATRIX
    data.dense_matrix.m, data.dense_matrix.n = value.shape
    data.dense_matrix.value_bytes = value.tobytes(order="Fortran")
    return data

def dense_matrix_metadata(value):
    data = data_pb2.Data()
    data.data_type = data_pb2.Data.DENSE_MATRIX
    data.dense_matrix.m, data.dense_matrix.n = value.shape
    return data

def value_location(value):
    return "/mem/data/" + str(abs(hash(value.tostring())))

def store_constant(value):
    assert isinstance(value, numpy.ndarray)
    prefix = value_location(value)

    if data.metadata_file(prefix) not in constants:
        data_map[data.metadata_file(prefix)] = (
            data.dense_matrix_metadata(value).SerializeToString())
        data_map[data.value_file(prefix)] = value.tobytes(order="F")

    return expression.constant(
        m=value.shape[0],
        n=value.shape[1],
        data_location=prefix)
