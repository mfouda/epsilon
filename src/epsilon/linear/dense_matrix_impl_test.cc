
#include <gtest/gtest.h>

#include "epsilon/vector/vector_testutil.h"
#include "epsilon/linear/dense_matrix_impl.h"

namespace linear_map {

class DenseMatrixImplTest : public testing::Test {
 protected:
  DenseMatrixImplTest() {
    srand(0);
    A0 = Eigen::MatrixXd::Random(2,3);
    A = LinearMap(new DenseMatrixImpl(A0));
    x = Eigen::VectorXd::Random(3);
    y = Eigen::VectorXd::Random(2);
  }

  Eigen::MatrixXd A0;
  LinearMap A;
  Eigen::VectorXd x, y;
};

TEST_F(DenseMatrixImplTest, Apply) {
  EXPECT_TRUE(VectorEquals(A0*x, A.impl().Apply(x), 1e-8));
  EXPECT_TRUE(
      VectorEquals(A0.transpose()*y,
                   A.Transpose().impl().Apply(y), 1e-8));
}

}  // namespace linear_map
