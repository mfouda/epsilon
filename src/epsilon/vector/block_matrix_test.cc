
#include <gtest/gtest.h>

#include "epsilon/vector/block_matrix.h"
#include "epsilon/vector/block_vector.h"
#include "epsilon/vector/vector_testutil.h"
#include "epsilon/vector/vector_util.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"

class BlockMatrixTest : public testing::Test {
 protected:
  BlockMatrixTest() : A0_(3,2), B0_(3,2) {
    A0_ << 1, 2, 3, 4, 5, 6;

    B0_.coeffRef(0,1) = 1;
    B0_.coeffRef(1,0) = -1;

    I_ = Eigen::MatrixXd::Identity(3,3);
  }
  Eigen::MatrixXd A0_;
  Eigen::MatrixXd I_;
  SparseXd B0_;
};

TEST_F(BlockMatrixTest, Assignment) {
  BlockMatrix A;
  A("row1", "col1") = LinearMap(new DenseMatrixImpl(A0_));
  EXPECT_TRUE(MatrixEquals(A0_, A("row1", "col1").impl().AsDense()));
  A("row1", "col1") = LinearMap(new SparseMatrixImpl(B0_));
  EXPECT_TRUE(MatrixEquals(B0_, A("row1", "col1").impl().AsDense()));
}

TEST_F(BlockMatrixTest, Transpose) {
  BlockMatrix A;
  A("0", "0") = LinearMap(new DenseMatrixImpl(A0_));
  A("0", "1") = LinearMap(new DenseMatrixImpl(I_));
  BlockMatrix AT = A.Transpose();

  EXPECT_TRUE(MatrixEquals(A0_.transpose(), AT("0", "0").impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(I_, AT("1", "0").impl().AsDense()));
}

TEST_F(BlockMatrixTest, MultiplyVector) {
  BlockMatrix A;
  A("0", "0") = LinearMap(new DenseMatrixImpl(A0_));
  A("0", "1") = LinearMap(new DenseMatrixImpl(I_));

  Eigen::VectorXd x0(2), x1(3);
  x0 << 1, -2;
  x1 << 10, 11, 12;
  BlockVector x;
  x("0") = x0;
  x("1") = x1;

  EXPECT_TRUE(VectorEquals(A0_*x0 + x1, (A*x)("0")));
}

TEST_F(BlockMatrixTest, MultiplyMatrix) {
  BlockMatrix A;
  A("0", "0") = LinearMap(new DenseMatrixImpl(A0_));
  A("0", "1") = LinearMap(new SparseMatrixImpl(B0_));

  BlockMatrix B;
  B("0", "3") = LinearMap(new DenseMatrixImpl(A0_.transpose()));
  B("1", "3") = LinearMap(new SparseMatrixImpl(
      static_cast<SparseXd>(B0_.transpose())));

  BlockMatrix C = A*B;
  Eigen::MatrixXd expected = (
      A0_*A0_.transpose() +
      static_cast<Eigen::MatrixXd>(
          (B0_*B0_.transpose()).eval()));
  EXPECT_TRUE(MatrixEquals(expected, C("0", "3").impl().AsDense()));
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
//   b("0") = Eigen::VectorXd(3);
//   b("0") << 1, 2, 3;

//   EXPECT_TRUE(VectorEquals(solver2.solve(b("0")), solver->solve(b)("0")));
// }
