
#include <gtest/gtest.h>

#include "epsilon/vector/block_vector.h"
#include "epsilon/vector/block_matrix.h"
#include "epsilon/vector/vector_testutil.h"

class BlockMatrixTest : public testing::Test {
 protected:
  BlockMatrixTest() : dense_(3,2), sparse_(3,2) {
    dense_ << 1, 2, 3, 4, 5, 6;

    sparse_.coeffRef(0,1) = 1;
    sparse_.coeffRef(1,0) = -1;
  }
  Eigen::MatrixXd dense_;
  SparseXd sparse_;
};

TEST_F(BlockMatrixTest, Assignment) {
  BlockMatrix A;
  A("row1", "col1") = MatrixVariant(dense_);
  EXPECT_TRUE(MatrixEquals(dense_, A("row1", "col1").AsDense()));
  A("row1", "col1") = MatrixVariant(sparse_);
  EXPECT_TRUE(MatrixEquals(sparse_, A("row1", "col1").AsDense()));
}

TEST_F(BlockMatrixTest, Transpose) {
  BlockMatrix A;
  A("0", "0") = MatrixVariant(dense_);
  A("0", "1") = MatrixVariant(Eigen::MatrixXd::Identity(3,3).eval());
  BlockMatrix AT = A.transpose();

  BlockMatrix B = A.transpose();
  EXPECT_TRUE(MatrixEquals(
      dense_.transpose(),
      B("0", "0").AsDense()));
  EXPECT_TRUE(MatrixEquals(
      Eigen::MatrixXd::Identity(3, 3),
      B("1", "0").AsDense()));
}

TEST_F(BlockMatrixTest, MultiplyVector) {
  BlockMatrix A;
  A("0", "0") = MatrixVariant(dense_);
  A("0", "1") = MatrixVariant(Eigen::MatrixXd::Identity(3,3).eval());

  Eigen::VectorXd x0(2), x1(3);
  x0 << 1, -2;
  x1 << 10, 11, 12;
  BlockVector x;
  x("0") = x0;
  x("1") = x1;

  EXPECT_TRUE(VectorEquals(dense_*x0 + x1, (A*x)("0")));
}

TEST_F(BlockMatrixTest, MultiplyMatrix) {
  BlockMatrix A;
  A("0", "0") = MatrixVariant(dense_);
  A("0", "1") = MatrixVariant(sparse_);

  BlockMatrix B;
  B("0", "3") = MatrixVariant(dense_.transpose());
  B("1", "3") = MatrixVariant(
      static_cast<SparseXd>(sparse_.transpose()));

  BlockMatrix C = A*B;

  Eigen::MatrixXd expected = (
      dense_*dense_.transpose() +
      static_cast<Eigen::MatrixXd>(
          (sparse_*sparse_.transpose()).eval()));
  EXPECT_TRUE(MatrixEquals(expected, C("0", "3").AsDense()));
}

// TEST_F(BlockMatrixTest, SolveSingleDense) {
//   Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3,3);
//   BlockMatrix A;
//   A("0", "0") = MatrixVariant(dense_);
//   A("0", "1") = MatrixVariant(I);
//   std::unique_ptr<BlockMatrix::Solver> solver = (A*A.transpose()).inv();

//   Eigen::LLT<MatrixXd> solver2;
//   solver2.compute(dense_*dense_.transpose() + I);

//   BlockVector b;
//   b("0") << 1, 2, 3;

//   EXPECT_TRUE(VectorEquals(solver2.solve(b("0")), solver->solve(b)("0")));
// }
