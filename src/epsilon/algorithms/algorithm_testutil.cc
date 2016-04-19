
#include "epsilon/algorithms/algorithm_testutil.h"

Eigen::VectorXd ComputeLS(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
  Eigen::LLT<Eigen::MatrixXd> solver;
  solver.compute(A.transpose()*A);
  return solver.solve(A.transpose()*b);
}
