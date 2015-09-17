#ifndef UTIL_VECTOR_TESTUTIL_H
#define UTIL_VECTOR_TESTUTIL_H

#include <Eigen/Dense>
#include <gtest/gtest.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;

bool VectorEqualsImpl(const VectorXd& a, const VectorXd& b, double tol);

testing::AssertionResult VectorEquals(
  const VectorXd& expected,
  const VectorXd& actual,
  double tol = 0);

testing::AssertionResult MatrixEquals(
  const MatrixXd& expected,
  const MatrixXd& actual,
  double tol = 0);

Eigen::VectorXd TestVector(const std::vector<double>& b);
Eigen::MatrixXd TestMatrix(const std::vector<std::vector<double>>& A);

#endif  // UTIL_VECTOR_TESTUTIL_H
