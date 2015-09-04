
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression.h"
#include "epsilon/expression/expression_testutil.h"
#include "epsilon/operators/affine.h"
#include "epsilon/util/vector.h"
#include "epsilon/util/vector_testutil.h"

class LinearExpressionOperatorTest : public testing::Test {
 protected:

  void ExpectBuildOperator(const Expression& expr, const MatrixXd& expected) {
    std::unique_ptr<OperatorImpl> op(BuildLinearExpressionOperator(expr, false));

    MatrixXd actual;
    op->ToMatrix(&actual);
    EXPECT_TRUE(MatrixEquals(expected, actual, 1e-4));
  }
};

void TestBuildAffineOperator(
    const Expression& expr,
    int n,
    const Eigen::MatrixXd& expected_A,
    const Eigen::VectorXd& expected_b) {
  const int m = GetDimension(expr);
  Eigen::MatrixXd A(m,n);
  Eigen::VectorXd b(m);
  BuildAffineOperator(expr, n, 0, &A, &b);
  EXPECT_TRUE(MatrixEquals(expected_A, A, 1e-3));
  EXPECT_TRUE(VectorEquals(expected_b, b, 1e-3));
}

TEST(BuildAffineOperator, VectorConstant) {
  TestBuildAffineOperator(
      TestConstant(TestVector({1,2,3})),
      0,
      TestMatrix({{},{},{}}),
      TestVector({1,2,3}));
}

TEST(BuildAffineOperator, MatrixConstant) {
  TestBuildAffineOperator(
      TestConstant(TestMatrix({{1,2,3},{4,5,6}})),
      0,
      TestMatrix({{},{},{},{},{},{}}),
      TestVector({1,4,2,5,3,6}));
}

TEST(BuildAffineOperator, IndexVectorConstant) {
  TestBuildAffineOperator(
      expression::Index(1, 2, TestConstant(TestVector({1,2,3}))),
      0,
      TestMatrix({{},{}}),
      TestVector({2,3}));
}

TEST(BuildAffineOperator, IndexMatrixConstant) {
  TestBuildAffineOperator(
      expression::Index(
          0, 1, 0, 2,
          TestConstant(TestMatrix({{1,2,3},{4,5,6}}))),
      0,
      TestMatrix({{},{}}),
      TestVector({1,2}));
}

// Test to make sure we're not doing anything dumb with memory
TEST(BuildAffineOperator, IndexMatrixConstant_Large) {
  const int m = 1000;
  const int n = 2000;
  Eigen::MatrixXd C = Eigen::MatrixXd::Constant(m, n, 1);
  TestBuildAffineOperator(
      expression::Index(0, 500, 1000, 500, TestConstant(C)),
      0,
      Eigen::MatrixXd(500*500, 0),
      Eigen::VectorXd::Constant(500*500, 1));
}

// Ax
TEST(BuildAffineOperator, MultiplyVectorVariable) {
  const int m = 3;
  const int n = 2;
  TestBuildAffineOperator(
      expression::Multiply(
          TestConstant(TestMatrix({{1,2},{3,4},{5,6}})),
          TestVariable(n, 1)),
      n,
      TestMatrix({{1,2},{3,4},{5,6}}),
      Eigen::VectorXd::Zero(m));
}

// AX
TEST(BuildAffineOperator, MultiplyMatrixVariable) {
  const int m = 4;
  const int n = 2;
  const int k = 3;
  TestBuildAffineOperator(
      expression::Multiply(
          TestConstant(TestMatrix({{1,2},{3,4},{5,6},{7,8}})),
          TestVariable(n, k)),
      n*k,
      BlockDiag(TestMatrix({{1,2},{3,4},{5,6},{7,8}}), k),
      Eigen::VectorXd::Zero(m*k));
}

TEST(BuildAffineOperator, HStack) {
  const int m = 3;
  const int n = 2;

  TestBuildAffineOperator(
      expression::HStack({
          expression::Variable(m, n, "x", 0),
          expression::Variable(m, n, "y", m*n)}),
      m*n*2,
      Eigen::MatrixXd::Identity(m*n*2, m*n*2),
      Eigen::VectorXd::Zero(m*n*2));
}

TEST(BuildAffineOperator, VStack) {
  const int m = 3;
  const int n = 2;

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m*n*2, m*n*2);
  A.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3,3);
  A.block(6, 3, 3, 3) = Eigen::MatrixXd::Identity(3,3);
  A.block(3, 6, 3, 3) = Eigen::MatrixXd::Identity(3,3);
  A.block(9, 9, 3, 3) = Eigen::MatrixXd::Identity(3,3);

  TestBuildAffineOperator(
      expression::VStack({
          expression::Variable(m, n, "x", 0),
          expression::Variable(m, n, "y", m*n)}),
      m*n*2, A, Eigen::VectorXd::Zero(m*n*2));
}
