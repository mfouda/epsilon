
import numpy as np
import scipy.sparse as sp

from epsilon.data_pb2 import Data
from epsilon import expression

# Global store of all constants
data_map = {}

METADATA_FILE = "metadata"
VALUE_FILE = "value"

def metadata_file(prefix):
    return prefix + "/" + METADATA_FILE

def value_file(prefix):
    return prefix + "/" + VALUE_FILE

def value_location(value):
    return "/mem/data/" + str(abs(hash(value)))

def value_data(value):
    # TODO(mwytock): Need to ensure type double/float here?
    if isinstance(value, np.ndarray):
        metadata = Data(
            data_type=Data.DENSE_MATRIX,
            m=value.shape[0],
            n=1 if len(value.shape) == 1 else value.shape[1])
        value_bytes = value.tobytes(order="F")

    elif isinstance(value, sp.spmatrix):
        csc = value.tocsc()
        metadata = Data(
            data_type=Data.SPARSE_MATRIX,
            m=value.shape[0],
            n=1 if len(value.shape) == 1 else value.shape[1],
            nnz=value.nnz)
        value_bytes = (csc.data.tobytes("F") +
                       csc.indices.tobytes("F") +
                       csc.indptr.tobytes("F"))

    else:
        raise ValueError("unknown value type " + str(value))

    return metadata, value_bytes

def store_constant(value):
    metadata, value_bytes = value_data(value)
    prefix = value_location(value_bytes)
    data_map[metadata_file(prefix)] = metadata.SerializeToString()
    data_map[value_file(prefix)] = value_bytes
    return expression.constant(
        metadata.m, metadata.m, data_location=prefix)
