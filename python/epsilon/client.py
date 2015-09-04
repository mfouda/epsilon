
import os
import logging
import time

from distopt import cvxpy_expr
from distopt import master_pb2
from distopt import problem_pb2
from distopt import solver_pb2

HOST = os.getenv("DISTOPT_HOST", "distopt.com")
PORT = 8000

TIMEOUT_SECS = 10.0
STATUS_DELAY_SECS = 1.0

# TODO(mwytock): These interfaces will be combined
master_stub = solver_pb2.early_adopter_create_SolverService_stub(HOST, PORT)
master_stub.__enter__()

master_stub2 = master_pb2.early_adopter_create_Master_stub(HOST, PORT)
master_stub2.__enter__()

def status_url(problem_id):
    return "http://%s:8080/problems/%s" % (HOST, problem_id)

def residuals_line(status):
    s = ""
    if status.HasField("residuals"):
        s = "residuals: %.2e [%.2e], %.2e [%.2e]" % (
            status.residuals.r_norm,
            status.residuals.epsilon_primal,
            status.residuals.s_norm,
            status.residuals.epsilon_dual)

    if status.HasField("cone_residuals"):
        s += "residuals: %.2e %.2e, gap: %2.e, " % (
            status.cone_residuals.primal_residual,
            status.cone_residuals.dual_residual,
            status.cone_residuals.duality_gap)
    return s

def status_line(status):
    obj = ("obj=%.2f, " % status.objective_value
           if status.objective_value else "")
    return "iter=%d, %s%s" % (status.num_iterations, obj, residuals_line(status))

def start(**kwargs):
    logging.info("Starting problem on %s", HOST)
    request = solver_pb2.StartRequest(**kwargs)

    # Default to Consensus-Prox algorithm
    if request.algorithm == solver_pb2.StartRequest.UNKNOWN:
        if len(request.prox_problem) > 1:
            request.algorithm = solver_pb2.StartRequest.CONSENSUS_PROX_DIST
        else:
            request.algorithm = solver_pb2.StartRequest.CONSENSUS_PROX

    response = master_stub.Start(request, TIMEOUT_SECS)
    logging.info("Started problem, status: %s", status_url(response.problem_id))

    return response.problem_id

def solve(**kwargs):
    """Start the problem and wait for it to complete."""
    problem_id = start(**kwargs)
    request = solver_pb2.GetStatusRequest(problem_id=problem_id)
    logging.info("Waiting for backend to start")
    printed_num_workers = False

    while True:
        response = master_stub.GetStatus(request, TIMEOUT_SECS)
        if (response.status.state != problem_pb2.ProblemStatus.NOT_STARTED and
            response.status.state != problem_pb2.ProblemStatus.INITIALIZING and
            response.status.state != problem_pb2.ProblemStatus.RUNNING):
            break

        if response.status.num_iterations != 0:
            if not printed_num_workers and response.status.num_workers:
                logging.info("Solving problem with %d workers",
                             response.status.num_workers)
                printed_num_workers = True
            logging.info(status_line(response.status))

        time.sleep(STATUS_DELAY_SECS)

    logging.info(status_line(response.status))
    logging.info(
        "Finished, state=%s",
        problem_pb2.ProblemStatus.State.Name(response.status.state))
    return problem_id

def evaluate(**kwargs):
    request = master_pb2.EvaluateRequest(**kwargs)
    response = master_stub2.Evaluate(request, TIMEOUT_SECS)
    return response.value
