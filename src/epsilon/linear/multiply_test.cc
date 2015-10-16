
#include <gtest/gtest.h>

#include "epsilon/linear/linear_map.h"
#include "epsilon/vector/vector_testutil.h"

TEST(MatrixVariantTest, Multiply) {
  Eigen::MatrixXd A0(2,2);
  Eigen::SparseMatrix<double> B0(2,2);
  A0 << 1, 2, 3, 4;
  B0.coeffRef(0,1) = 1;
  B0.coeffRef(1,0) = -1;

  LinearMap A = LinearMap::FromDense(A0);
  LinearMap B = LinearMap::FromSparse(B0);

  EXPECT_TRUE(MatrixEquals(A0*A0, (A*A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(A0*B0, (A*B).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals(B0*A0, (B*A).impl().AsDense()));
  EXPECT_TRUE(MatrixEquals((B0*B0).eval(), (B*B).impl().AsDense()));
}
