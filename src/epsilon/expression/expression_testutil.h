#ifndef EXPRESSION_EXPRESSION_TESTUTIL_H
#define EXPRESSION_EXPRESSION_TESTUTIL_H

#include <Eigen/Dense>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "epsilon/expression.pb.h"

Eigen::MatrixXd RandomConstant(int m, int n, Expression* expr);
Expression RandomConstantOp(
    int m, int n, const std::string& input_key, Eigen::MatrixXd* A);

Expression TestConstant(const Eigen::MatrixXd& A);
Expression TestVariable(int m, int n);

#endif  // EXPRESSION_EXPRESSION_TESTUTIL_H
