
from client import *
from data import *
from distributed_file import *
from expression import *
from matrix import Matrix, Vector, write_matrix

# Hack to making specifying algorithms easier
# TODO(mwytock): Move the algorithm enum into a more reasonable place
CONSENSUS_PROX = solver_pb2.StartRequest.CONSENSUS_PROX
CONSENSUS_PROX_DIST = solver_pb2.StartRequest.CONSENSUS_PROX_DIST
CONSENSUS_EPSILON = solver_pb2.StartRequest.CONSENSUS_EPSILON
