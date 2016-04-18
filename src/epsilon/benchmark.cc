
#include <glog/logging.h>

#include "epsilon/expression.pb.h"
#include "epsilon/solver_params.pb.h"
#include "epsilon/util/file.h"
#include "epsilon/algorithms/prox_admm.h"

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  Problem problem;
  CHECK(problem.ParseFromString(ReadStringFromFile(argv[1])));

  SolverParams params;
  params.set_rel_tol(1e-3);
  ProxADMMSolver solver(problem, params);
  solver.Solve();
}
