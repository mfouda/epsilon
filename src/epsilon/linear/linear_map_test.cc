
#include <gtest/gtest.h>

#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/diagonal_matrix_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"
#include "epsilon/vector/vector_testutil.h"

class LinearMapTest : public testing::Test {
 protected:
  LinearMapTest() {
    A0 = Eigen::MatrixXd(2,2);
    A0 << 1, 2, 3, 4;
    A = LinearMap(new DenseMatrixImpl(A0));

    Eigen::SparseMatrix<double> sparse(2,2);
    sparse.coeffRef(0,1) = 1;
    sparse.coeffRef(1,0) = -1;
    B0 = static_cast<Eigen::MatrixXd>(sparse);
    B = LinearMap(new SparseMatrixImpl(sparse));

    const double c = -3.2;
    C0 = c*Eigen::MatrixXd::Identity(2,2);
    C = LinearMap(new ScalarMatrixImpl(2,c));

    Eigen::DiagonalMatrix<double, Eigen::Dynamic> diag(2);
    diag.diagonal() << -1, 3;
    D0 = static_cast<Eigen::MatrixXd>(diag);
    D = LinearMap(new DiagonalMatrixImpl(diag));

    x = Eigen::VectorXd(2);
    x << 3,4;
  }

  Eigen::VectorXd x;
  Eigen::MatrixXd A0, B0, C0, D0;
  LinearMap A, B, C, D;
};

TEST_F(LinearMapTest, Multiply) {
  EXPECT_TRUE(MatrixEquals(A0*A0, (A*A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(A0*B0, (A*B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(A0*C0, (A*C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(A0*D0, (A*D).impl().AsDense()));

  EXPECT_TRUE(MatrixEquals(B0*A0, (B*A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(B0*B0, (B*B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(B0*C0, (B*C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(B0*D0, (B*D).impl().AsDense()));

  EXPECT_TRUE(MatrixEquals(C0*A0, (C*A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(C0*B0, (C*B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(C0*C0, (C*C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(C0*D0, (C*D).impl().AsDense()));

  EXPECT_TRUE(MatrixEquals(D0*A0, (D*A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(D0*B0, (D*B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(D0*C0, (D*C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(D0*D0, (D*D).impl().AsDense()));
}

TEST_F(LinearMapTest, Add) {
  EXPECT_TRUE(MatrixEquals(A0+A0, (A+A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(A0+B0, (A+B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(A0+C0, (A+C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(A0+D0, (A+D).impl().AsDense()));

  EXPECT_TRUE(MatrixEquals(B0+A0, (B+A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(B0+B0, (B+B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(B0+C0, (B+C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(B0+D0, (B+D).impl().AsDense()));

  EXPECT_TRUE(MatrixEquals(C0+A0, (C+A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(C0+B0, (C+B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(C0+C0, (C+C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(C0+D0, (C+D).impl().AsDense()));

  EXPECT_TRUE(MatrixEquals(D0+A0, (D+A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(D0+B0, (D+B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(D0+C0, (D+C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(D0+D0, (D+D).impl().AsDense()));
}

TEST_F(LinearMapTest, Apply) {
  EXPECT_TRUE(VectorEquals(A0*x, A*x));
  EXPECT_TRUE(VectorEquals(B0*x, B*x));
  EXPECT_TRUE(VectorEquals(C0*x, C*x));
  EXPECT_TRUE(VectorEquals(D0*x, D*x));
}
