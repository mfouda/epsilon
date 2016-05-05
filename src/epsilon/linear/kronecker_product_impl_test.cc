
#include <gtest/gtest.h>

#include "epsilon/vector/vector_testutil.h"
#include "epsilon/linear/kronecker_product_impl.h"

namespace linear_map {

TEST(KroneckerProductImplTest, Apply) {
  srand(0);
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(2,3);
  Eigen::MatrixXd B = Eigen::MatrixXd::Random(4,5);
  LinearMap C(new KroneckerProductImpl(
      LinearMap(new DenseMatrixImpl(A)),
      LinearMap(new DenseMatrixImpl(B))));

  Eigen::MatrixXd X = Eigen::MatrixXd::Random(5,3);
  EXPECT_TRUE(VectorEquals(
      ToVector(B*X*A.transpose()), C.impl().Apply(ToVector(X)), 1e-8));
}

}  // namespace linear_map
