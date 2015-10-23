
import numpy as np
import scipy.sparse as sp

from epsilon.data_pb2 import Data
from epsilon import expression

METADATA_FILE = "metadata"
VALUE_FILE = "value"

data_map = {}
expr_map = {}

def metadata_file(prefix):
    return prefix + "/" + METADATA_FILE

def value_file(prefix):
    return prefix + "/" + VALUE_FILE

def value_location(value):
    # TODO(mwytock): Better hash function here? Some matrices may have same
    # tostring() but different values.
    return "/mem/data/" + str(abs(hash(value.tostring())))

def store_ndarray(value, m, n, prefix):
    data_map[metadata_file(prefix)] = Data(
        data_type=Data.DENSE_MATRIX, m=m, n=n).SerializeToString()
    data_map[value_file(prefix)] = value.tobytes(order="F")

def store_coo_matrix(value, m, n, prefix):
    data_map[metadata_file(prefix)] = Data(
        data_type=Data.SPARSE_MATRIX, m=m, n=n).SerializeToString()
    data_map[value_file(prefix)] = 0

def store_constant(value):
    prefix = value_location(value)

    expr = expr_map.get(prefix, None)
    if not expr:
        m = value.shape[0]
        n = 1 if len(value.shape) == 1 else value.shape[1]

        if isinstance(value, np.ndarray):
             store_ndarray(value, m, n, prefix)
        elif isinstance(value, sp.coo_matrix):
             store_coo_matrix(value, m, n, prefix)
        else:
            raise ValueError("unknown value: " + str(value))

        expr = expression.constant(m, n, data_location=prefix)
        expr_map[prefix] = expr

    return expr
