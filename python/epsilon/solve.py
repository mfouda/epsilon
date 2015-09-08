
from epsilon import _solve
from epsilon import cvxpy_expr
from epsilon import prox
from epsilon import status_pb2
from epsilon import solver_params_pb2

def solve(problem, params=solver_params_pb2.SolverParams()):
    prob_proto, data = cvxpy_expr.convert_problem(prob)
    prox_prob_proto = prox.convert_problem(prob_proto)
    return status_pb2.ProblemStatus.FromString(
        _solve.prox_admm_solve(
            prox_proto.SerializeToString(),
            params.SerializeToString(),
            data))
