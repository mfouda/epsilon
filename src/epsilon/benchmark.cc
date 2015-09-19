
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "epsilon/expression.pb.h"
#include "epsilon/parameters/local_parameter_service.h"
#include "epsilon/solver_params.pb.h"
#include "epsilon/util/file.h"
#include "epsilon/algorithms/prox_admm.h"

DEFINE_string(problem, "", "");

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  Problem problem;
  CHECK(problem.ParseFromString(ReadStringFromFile(FLAGS_problem)));

  SolverParams params;
  ProxADMMSolver solver(
      problem, params,
      std::unique_ptr<ParameterService>(new LocalParameterService));
  solver.Solve();
}
