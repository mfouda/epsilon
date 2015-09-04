#ifndef DISTOPT_ALGORITHMS_ALGORITHM_TESTUTIL_H
#define DISTOPT_ALGORITHMS_ALGORITHM_TESTUTIL_H

#include <Eigen/Dense>

#include "distopt/problem.pb.h"

Eigen::VectorXd ComputeLS(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

Problem DenseLassoProblem(int m, int n, double p);

Eigen::VectorXd FetchRemoteParameter(uint64_t parameter_id);

#endif  // DISTOPT_ALGORITHMS_ALGORITHM_TESTUTIL_H
