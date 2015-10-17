
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "epsilon/affine/affine.h"
#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression.h"
#include "epsilon/expression/expression_testutil.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"
#include "epsilon/vector/vector_testutil.h"
#include "epsilon/vector/vector_util.h"

class BuildAffineOperatorTest : public testing::Test {
 protected:
  void Test(const Expression& expr) {
    BlockMatrix A1;
    BlockVector b1;
    affine::BuildAffineOperator(expr, "_", &A1, &b1);
    for (const std::string& col_key : A0.col_keys()) {
      EXPECT_TRUE(MatrixEquals(
          A0("_", col_key).impl().AsDense(),
          A1("_", col_key).impl().AsDense(), 1e-3));
    }

    if (!b0.has_key("_"))
      EXPECT_FALSE(b1.has_key("_"));
    else
      EXPECT_TRUE(VectorEquals(b0("_"), b1("_"), 1e-3));
  }

  BlockMatrix A0;
  BlockVector b0;
};

TEST_F(BuildAffineOperatorTest, VectorConstant) {
  b0("_") = TestVector({1,2,3});
  Test(TestConstant(TestVector({1,2,3})));
}

TEST_F(BuildAffineOperatorTest, MatrixConstant) {
  b0("_") = TestVector({1,4,2,5,3,6});
  Test(TestConstant(TestMatrix({{1,2,3},{4,5,6}})));
}

TEST_F(BuildAffineOperatorTest, IndexVectorConstant) {
  b0("_") = TestVector({2,3});
  Test(expression::Index(1, 2, TestConstant(TestVector({1,2,3}))));
}

TEST_F(BuildAffineOperatorTest, IndexMatrixConstant) {
  b0("_") = TestVector({1,2});
  Test(expression::Index(
      0, 1, 0, 2,
      TestConstant(TestMatrix({{1,2,3},{4,5,6}}))));
}

// Test to make sure we're not doing anything dumb with memory
TEST_F(BuildAffineOperatorTest, IndexMatrixConstant_Large) {
  const int m = 1000;
  const int n = 2000;
  Eigen::MatrixXd C = Eigen::MatrixXd::Constant(m, n, 1);
  b0("_") = Eigen::VectorXd::Constant(500*500, 1);
  Test(expression::Index(0, 500, 1000, 500, TestConstant(C)));
}

// Ax
TEST_F(BuildAffineOperatorTest, MultiplyVectorVariable) {
  const int n = 2;
  A0("_", "x") = LinearMap(new DenseMatrixImpl(
      TestMatrix({{1,2},{3,4},{5,6}})));
  Test(expression::Multiply(
      TestConstant(TestMatrix({{1,2},{3,4},{5,6}})),
      expression::Variable(n, 1, "x")));
}

// AX
TEST_F(BuildAffineOperatorTest, MultiplyMatrixVariable) {
  const int n = 2;
  const int k = 3;

  A0("_", "x") = LinearMap(new SparseMatrixImpl(
      BlockDiag(TestMatrix({{1,2},{3,4},{5,6},{7,8}}), k)));
  Test(expression::Multiply(
      TestConstant(TestMatrix({{1,2},{3,4},{5,6},{7,8}})),
      expression::Variable(n, k, "x")));
}

// TEST(BuildAffineOperator, HStack) {
//   const int m = 3;
//   const int n = 2;

//   TestBuildAffineOperator(
//       expression::HStack({
//           expression::Variable(m, n, "x"),
//           expression::Variable(m, n, "y")}),
//       Eigen::MatrixXd::Identity(m*n*2, m*n*2),
//       Eigen::VectorXd::Zero(m*n*2));
// }

// TEST(BuildAffineOperator, HStack_Offset) {
//   const int m = 3;
//   const int n = 2;

//   Expression hstack = expression::HStack({expression::Variable(m, n, "y")});
//   hstack.mutable_stack_params()->set_offset(2);
//   *hstack.mutable_size() = CreateSize(3, 4);

//   Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m*n*2, m*n);
//   A.block(m*n, 0, m*n, m*n) = Eigen::MatrixXd::Identity(m*n, m*n);
//   TestBuildAffineOperator(hstack, A, Eigen::VectorXd::Zero(m*n*2));
// }

// TEST(BuildAffineOperator, VStack) {
//   const int m = 3;
//   const int n = 2;

//   Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m*n*2, m*n*2);
//   A.block(0, 0, m, m) = Eigen::MatrixXd::Identity(3,3);
//   A.block(m*n, m, m, m) = Eigen::MatrixXd::Identity(3,3);
//   A.block(m, m*n, m, m) = Eigen::MatrixXd::Identity(3,3);
//   A.block(m*m, m*m, m, m) = Eigen::MatrixXd::Identity(3,3);

//   TestBuildAffineOperator(
//       expression::VStack({
//           expression::Variable(m, n, "x"),
//           expression::Variable(m, n, "y")}),
//       A, Eigen::VectorXd::Zero(m*n*2));
// }

// TEST(BuildAffineOperator, VStack_Offset) {
//   const int m = 3;
//   const int n = 2;

//   Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m*n*2, m*n);
//   A.block(m, 0, m, m) = Eigen::MatrixXd::Identity(m,m);
//   A.block(m*m, m, m, m) = Eigen::MatrixXd::Identity(m,m);

//   Expression vstack = expression::VStack({expression::Variable(m, n, "y")});
//   vstack.mutable_stack_params()->set_offset(3);
//   *vstack.mutable_size() = CreateSize(6, 2);
//   TestBuildAffineOperator(vstack, A, Eigen::VectorXd::Zero(m*n*2));
// }

TEST(GetProjection, Basic) {
  VariableOffsetMap a;
  Expression x = expression::Variable(4, 1, "x");
  Expression y = expression::Variable(3, 1, "y");
  a.Insert(x);
  a.Insert(y);

  {
    VariableOffsetMap b;
    b.Insert(x);
    SparseXd P = GetProjection(a, b);
    Eigen::MatrixXd expected_P = Eigen::MatrixXd::Zero(4, 7);
    expected_P.block(0, 0, 4, 4) += Eigen::MatrixXd::Identity(4, 4);
    EXPECT_TRUE(MatrixEquals(expected_P, P));
  }

  {
    VariableOffsetMap b;
    b.Insert(y);
    SparseXd P = GetProjection(a, b);
    Eigen::MatrixXd expected_P = Eigen::MatrixXd::Zero(3, 7);
    expected_P.block(0, 4, 3, 3) += Eigen::MatrixXd::Identity(3, 3);
    EXPECT_TRUE(MatrixEquals(expected_P, P));
  }

}
