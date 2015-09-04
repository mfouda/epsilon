#ifndef EPSILON_ALGORITHMS_ALGORITHM_TESTUTIL_H
#define EPSILON_ALGORITHMS_ALGORITHM_TESTUTIL_H

#include <Eigen/Dense>

Eigen::VectorXd ComputeLS(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

#endif  // EPSILON_ALGORITHMS_ALGORITHM_TESTUTIL_H
