#ifndef EPSILON_UTIL_VECTOR_FILE_H
#define EPSILON_UTIL_VECTOR_FILE_H

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "epsilon/expression.pb.h"

// Read path is broken up in to two steps currently
Eigen::MatrixXd GetMatrixData(const Constant& constant);

#endif  // EPSILON_UTIL_VECTOR_FILE_H
