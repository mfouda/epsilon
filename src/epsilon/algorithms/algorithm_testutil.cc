
#include <grpc++/channel_arguments.h>
#include <grpc++/create_channel.h>

#include "epsilon/algorithms/algorithm_testutil.h"
#include "epsilon/expression/expression_testutil.h"

Eigen::VectorXd ComputeLS(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
  Eigen::LLT<Eigen::MatrixXd> solver;
  solver.compute(A.transpose()*A);
  return solver.solve(A.transpose()*b);
}
