#!/usr/bin/env python

import logging
import argparse
import time

from google.protobuf import text_format
import cvxpy
import numpy

import distopt
import distopt.prox
from distopt import solver_params_pb2
from distopt import solver_pb2

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
# run_demo parameters
parser.add_argument("problem")
parser.add_argument("args", nargs="*")
parser.add_argument("--print", dest="print_prob", action="store_true",
                    help="print problem in expression tree format")
parser.add_argument("--distributed", action="store_true")

parser.add_argument("--scs", action="store_true",
                    help="also solve the problem locally")
parser.add_argument("--scs-indirect", action="store_true")
parser.add_argument("--scs-max-iters", default=2500, type=int)
parser.add_argument("--params")
args = parser.parse_args()

def parse_params():
    if not args.params:
        return solver_params_pb2.SolverParams()
    return text_format.Parse(args.params, solver_params_pb2.SolverParams())

def main():
    spec = __import__(args.problem)
    numpy.random.seed(0)
    prob= spec.create(*args.args)

    if args.scs:
        t0 = time.time()
        prob.solve(
            solver=cvxpy.SCS, verbose=True, use_indirect=args.scs_indirect,
            max_iters=args.scs_max_iters)
        t1 = time.time()
        print "Local solve time: %.2f seconds" % (t1-t0)
        print "Objective value:", prob.objective.value
        return

    prob_proto, data_proto = distopt.cvxpy_expr.convert_problem(prob)
    prox_protos = distopt.prox.convert_problem(
        prob_proto, distributed=args.distributed)

    if args.print_prob:
        print "Expression input:"
        print distopt.problem_str(prob_proto)
        print

        for i, prox_proto in enumerate(prox_protos):
            print "Prox problem:", i
            print distopt.prox_problem_str(prox_proto)

        return

    t0 = time.time()
    problem_id = distopt.solve(
        prox_problem=prox_protos,
        params=parse_params(),
        inline_data=data_proto)
    t1 = time.time()
    print "Solve time: %.2f seconds" % (t1-t0)

    # TODO(mwytock): Implement evaluation of large problems
    if not args.distributed:
        obj_val = distopt.evaluate(
            problem_id=problem_id,
            expression=prob_proto.objective)
        assert obj_val.m == 1 and obj_val.n == 1
        print "Objective value:", obj_val.value[0]

if __name__ == "__main__":
    main()
