
#include <gtest/gtest.h>

#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/diagonal_matrix_impl.h"
#include "epsilon/linear/kronecker_product_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"
#include "epsilon/vector/vector_testutil.h"

namespace linear_map {

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

    Eigen::MatrixXd E1(1,2), E2(2,1);
    E1 << 1, 2;
    E2 << 3, 4;
    E = LinearMap(new KroneckerProductImpl(
        LinearMap(new DenseMatrixImpl(E1)),
        LinearMap(new DenseMatrixImpl(E2))));
    E0 = E.impl().AsDense();

    x = Eigen::VectorXd(2);
    x << 3,4;
  }

  Eigen::VectorXd x;
  Eigen::MatrixXd A0, B0, C0, D0, E0;
  LinearMap A, B, C, D, E;
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

TEST_F(LinearMapTest, Multiply_KroneckerScalar) {
  EXPECT_TRUE(MatrixEquals(C0*E0, (C*E).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(E0*C0, (E*C).impl().AsDense()));
}

TEST_F(LinearMapTest, Add_KroneckerScalar) {
  LinearMap K(new KroneckerProductImpl(
      LinearMap(new DenseMatrixImpl(A0)),
      LinearMap(new ScalarMatrixImpl(3, -2.3))));
  LinearMap L(new KroneckerProductImpl(
      LinearMap(new ScalarMatrixImpl(3, 1.5)),
      LinearMap(new DenseMatrixImpl(A0))));
  LinearMap S(new ScalarMatrixImpl(6, 1.3));

  Eigen::MatrixXd K0 = K.impl().AsDense();
  Eigen::MatrixXd L0 = L.impl().AsDense();
  Eigen::MatrixXd S0 = S.impl().AsDense();

  EXPECT_TRUE(MatrixEquals(K0+S0, (K+S).impl().AsDense(), 1e-8));
  EXPECT_TRUE(MatrixEquals(S0+K0, (S+K).impl().AsDense(), 1e-8));
  EXPECT_TRUE(MatrixEquals(L0+S0, (L+S).impl().AsDense(), 1e-8));
  EXPECT_TRUE(MatrixEquals(S0+L0, (S+L).impl().AsDense(), 1e-8));
}

}  // namespace linear_map
