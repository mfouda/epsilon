
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

// TEST_F(BlockMatrixTest, MultiplyMatrix) {
//   BlockMatrix A;
//   A("0", "0") = dense_;
//   A("0", "1") = sparse_;

//   BlockMatrix B;
//   B("0", "3") = dense_.transpose();
//   B("1", "3") = sparse_.transpose();

//   // TODO(mwytock): Check result?
// }

// TEST_F(BlockMatrixTest, Transpose) {
//   BlockMatrix A;
//   A("0", "0") = A_;
//   A("0", "1") = Eigen::MatrixXd::Identity(3,3);
//   BlockMatrix AT = A.transpose();

//   // TODO(mwytock): Check result?
// }

// TEST_F(BlockMatrixTest, Inverse) {
//   Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3,3);
//   BlockMatrix A;
//   A("0", "0") = A_;
//   A("0", "1") = I;
//   BlockMatrix::Solver solver = (A*A.transpose()).inv();

//   Eigen::LLT<MatrixXd> solver2;
//   solver2.compute(A_*A_.transpose() + I);

//   BlockVector b;
//   b("0") << 1, 2, 3;

//   EXPECT_TRUE(VectorEquals(solver2.solve(b), solver.solve(b)));
// }
