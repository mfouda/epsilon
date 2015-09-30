
#include <gtest/gtest.h>

#include "epsilon/affine/affine_matrix.h"
#include "epsilon/expression/expression.h"
#include "epsilon/expression/expression_testutil.h"
#include "epsilon/vector/vector_testutil.h"
#include "epsilon/vector/vector_util.h"

// AX - B
TEST(AffineMatrix, MatrixMultiply) {
  Eigen::MatrixXd A = TestMatrix({{1, 2}, {3, 4}, {5, 6}});
  Eigen::MatrixXd B = TestMatrix(
      {{7, 8, 9}, {10, 11, 12}, {13, 14, 15}});
  Expression X = expression::Variable(2, 3, "X");

  Expression expr = expression::Add(
      expression::Multiply(TestConstant(A), X),
      expression::Negate(TestConstant(B)));

  affine::MatrixOperator op = affine::BuildMatrixOperator(expr);
  EXPECT_TRUE(MatrixEquals(A, op.A));
  EXPECT_TRUE(op.B.isIdentity());
  EXPECT_TRUE(MatrixEquals(-B, op.C));
}
