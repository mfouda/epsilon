#include "epsilon/util/vector_testutil.h"

#include <glog/logging.h>
#include "epsilon/util/vector.h"

using Eigen::Infinity;

bool VectorEqualsImpl(const VectorXd& a, const VectorXd& b, double tol) {
  return (a.rows() == b.rows() &&
	  (a - b).lpNorm<Infinity>() <= tol);
}

testing::AssertionResult VectorEquals(
  const VectorXd& expected,
  const VectorXd& actual,
  double tol) {
  if (VectorEqualsImpl(expected, actual, tol))
    return testing::AssertionSuccess();
  return testing::AssertionFailure() << VectorDebugString(expected) << " != "
                                     << VectorDebugString(actual);
}

testing::AssertionResult MatrixEquals(
  const MatrixXd& expected,
  const MatrixXd& actual,
  double tol) {
  if (expected.rows() == actual.rows() &&
      expected.cols() == actual.cols() &&
      (expected.rows() == 0 || expected.cols() == 0 ||
       (expected - actual).lpNorm<Infinity>() <= tol)) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure()
      << "\n" << MatrixDebugString(expected) << " != "
      << "\n" << MatrixDebugString(actual);
}

Eigen::VectorXd TestVector(const std::vector<double>& b) {
  return Eigen::Map<const Eigen::VectorXd>(&b[0], b.size());
}

Eigen::MatrixXd TestMatrix(const std::vector<std::vector<double>>& A_in) {
  if (A_in.size() == 0)
    return Eigen::MatrixXd(0,0);

  MatrixXd A(A_in.size(), A_in[0].size());
  for (int i = 0; i < A.rows(); i++) {
    A.row(i) = TestVector(A_in[i]);
  }
  return A;
}
