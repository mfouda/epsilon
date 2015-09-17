
#include <gtest/gtest.h>

#include "epsilon/vector/vector_util.h"

TEST(IsBlockScalar, Identity) {
  EXPECT_TRUE(IsBlockScalar(SparseIdentity(3)));
}

TEST(IsBlockScalar, Zero) {
  EXPECT_TRUE(IsBlockScalar(SparseXd(3, 3)));
}

TEST(IsBlockScalar, True) {
  SparseXd A(11, 3);
  A.insert(0, 0) = 1;
  A.insert(1, 1) = 1;
  A.insert(2, 2) = 1;
  EXPECT_TRUE(IsBlockScalar(A));
}

TEST(IsBlockScalar, DifferentValues) {
  SparseXd A(3, 3);
  A.insert(0, 0) = 1;
  A.insert(1, 1) = 2;
  A.insert(2, 2) = 2;
  EXPECT_FALSE(IsBlockScalar(A));
}

TEST(IsBlockScalar, ZeroFirstColumn) {
  SparseXd A(3, 3);
  A.insert(0, 1) = 1;
  EXPECT_FALSE(IsBlockScalar(A));
}
