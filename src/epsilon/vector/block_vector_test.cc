
#include <gtest/gtest.h>

#include "epsilon/vector/block_vector.h"
#include "epsilon/vector/vector_testutil.h"

class BlockVectorTest : public testing::Test {
 protected:
  BlockVectorTest() : a_(Eigen::VectorXd(3)), b_(Eigen::VectorXd(2)) {
    a_ << 1, 2, 3;
    b_ << 4, 5;
  }
  Eigen::VectorXd a_, b_;
};

TEST_F(BlockVectorTest, Assignment) {
  BlockVector v;
  v("a") = a_;
  EXPECT_TRUE(VectorEquals(a_, v("a")));
  v("a") = b_;
  EXPECT_TRUE(VectorEquals(b_, v("a")));
}

TEST_F(BlockVectorTest, AddSubtract) {
  BlockVector v, u, w;
  v("a") = a_;
  u("a") = 2*a_;
  u("b") = b_;

  w += v;
  EXPECT_TRUE(VectorEquals(a_, w("a")));
  EXPECT_TRUE(VectorEquals(3*a_, (v + u)("a")));
  EXPECT_TRUE(VectorEquals(b_, (v + u)("b")));

  w = v - u;
  EXPECT_TRUE(VectorEquals(-a_, w("a")));
  EXPECT_TRUE(VectorEquals(-b_, w("b")));
}
