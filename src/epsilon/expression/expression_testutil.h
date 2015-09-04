#ifndef EXPRESSION_EXPRESSION_TESTUTIL_H
#define EXPRESSION_EXPRESSION_TESTUTIL_H

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "distopt/expression/linear.h"
#include "distopt/problem.pb.h"

MatrixXd RandomConstant(int m, int n, Expression* expr);
Expression RandomConstantOp(
    int m, int n, const std::string& input_key, MatrixXd* A);

Expression TestConstant(const MatrixXd& A);
Expression TestVariable(int m, int n);

#endif  // EXPRESSION_EXPRESSION_TESTUTIL_H
