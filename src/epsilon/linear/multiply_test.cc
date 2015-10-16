
#include <gtest/gtest.h>

#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/diagonal_matrix_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"
#include "epsilon/vector/vector_testutil.h"

TEST(MatrixVariantTest, Multiply) {
  Eigen::MatrixXd A0(2,2);
  Eigen::SparseMatrix<double> B0(2,2);
  A0 << 1, 2, 3, 4;
  B0.coeffRef(0,1) = 1;
  B0.coeffRef(1,0) = -1;

  double c0 = -3.2;
  Eigen::MatrixXd C0 = c0*Eigen::MatrixXd::Identity(2,2);

  Eigen::DiagonalMatrix<double, Eigen::Dynamic> D0(2);
  D0.diagonal() << -1, 3;

  LinearMap A(new DenseMatrixImpl(A0));
  LinearMap B(new SparseMatrixImpl(B0));
  LinearMap C(new ScalarMatrixImpl(2,c0));
  LinearMap D(new DiagonalMatrixImpl(D0));

  Eigen::MatrixXd B0_B0 = (B0*B0).eval();
  Eigen::MatrixXd D0_D0 = Eigen::MatrixXd::Zero(2, 2);
  D0_D0.diagonal().array() = D0.diagonal().array() * D0.diagonal().array();

  EXPECT_TRUE(MatrixEquals(A0*A0, (A*A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(A0*B0, (A*B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(A0*C0, (A*C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(A0*D0, (A*D).impl().AsDense()));

  EXPECT_TRUE(MatrixEquals(B0*A0, (B*A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(B0_B0, (B*B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(B0*C0, (B*C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(B0*D0, (B*D).impl().AsDense()));

  EXPECT_TRUE(MatrixEquals(C0*A0, (C*A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(C0*B0, (C*B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(C0*C0, (C*C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(C0*D0, (C*D).impl().AsDense()));

  EXPECT_TRUE(MatrixEquals(D0*A0, (D*A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(D0*B0, (D*B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(D0*C0, (D*C).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(D0_D0, (D*D).impl().AsDense()));
}
