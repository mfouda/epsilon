
import numpy

from epsilon import _solve
from epsilon import cvxpy_expr
from epsilon import solver_pb2
from epsilon import solver_params_pb2

def solve(prob, params=solver_params_pb2.SolverParams()):
    prob_proto, data = cvxpy_expr.convert_problem(prob)
    prox_prob_proto = prox.convert_problem(prob_proto)

    status_str, values = _solve.prox_admm_solve(
        prox_prob_proto.SerializeToString(),
        params.SerializeToString(),
        data)

    for var in prob.variables():
        var_id = cvxpy_expr.variable_id(var)
        assert var_id in values
        x = numpy.fromstring(values[var_id], dtype=numpy.double)
        var.value = x.reshape(var.size[1], var.size[0]).transpose()

    return status_pb2.ProblemStatus.FromString(status_str)
