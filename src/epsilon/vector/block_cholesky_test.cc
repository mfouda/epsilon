
#include <gtest/gtest.h>

#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/vector/block_cholesky.h"
#include "epsilon/vector/block_matrix.h"
#include "epsilon/vector/block_vector.h"
#include "epsilon/vector/vector_testutil.h"
#include "epsilon/vector/vector_util.h"

// Declarations for fine-grained testing
BlockVector ForwardSub(
    const BlockMatrix& L,
    const std::vector<std::string>& keys,
    BlockVector b);
BlockVector BackSub(
    const BlockMatrix& L,
    const std::vector<std::string>& keys,
    BlockVector b);
int ComputeFill(const BlockMatrix& A, const std::string& k);

TEST(ForwardSub, Basic) {
  srand(0);
  BlockMatrix L;
  Eigen::MatrixXd L0 = Eigen::MatrixXd::Random(5, 2);
  L("two", "one") = linear_map::LinearMap(new linear_map::DenseMatrixImpl(L0));

  Eigen::VectorXd b1 = Eigen::VectorXd::Random(2);
  Eigen::VectorXd b2 = Eigen::VectorXd::Random(5);
  BlockVector b;
  b("one") = b1;
  b("two") = b2;
  BlockVector x = ForwardSub(L, {"one", "two"}, b);

  EXPECT_TRUE(VectorEquals(b1, x("one")));
  EXPECT_TRUE(VectorEquals(b2 - L0*b1, x("two")));
}

TEST(BackSub, Basic) {
  srand(0);
  BlockMatrix L;
  Eigen::MatrixXd L0 = Eigen::MatrixXd::Random(5, 2);
  L("two", "one") = linear_map::LinearMap(new linear_map::DenseMatrixImpl(L0));

  Eigen::VectorXd b1 = Eigen::VectorXd::Random(2);
  Eigen::VectorXd b2 = Eigen::VectorXd::Random(5);
  BlockVector b;
  b("one") = b1;
  b("two") = b2;
  BlockVector x = BackSub(L.Transpose(), {"one", "two"}, b);

  EXPECT_TRUE(VectorEquals(b1 - L0.transpose()*b2, x("one")));
  EXPECT_TRUE(VectorEquals(b2, x("two")));
}

TEST(ComputeFill, Basic) {
  srand(0);
  BlockMatrix A;

  // Matrix is [I A; A' I]
  Eigen::MatrixXd A0 = Eigen::MatrixXd::Random(5, 2);
  A("one", "one") = linear_map::Identity(5);
  A("one", "two") = linear_map::LinearMap(new linear_map::DenseMatrixImpl(A0));
  A("two", "one") = linear_map::LinearMap(
      new linear_map::DenseMatrixImpl(A0.transpose()));
  A("two", "two") = linear_map::Identity(2);

  EXPECT_EQ(4, ComputeFill(A, "one"));
  EXPECT_EQ(25, ComputeFill(A, "two"));
}

TEST(BlockCholesky, Basic) {
  srand(0);
  BlockMatrix A;

  // Matrix is [I A; A' I] and positive definite
  Eigen::MatrixXd A12 = Eigen::MatrixXd::Random(5, 2);
  A("one", "one") = linear_map::Scalar(10, 5);
  A("one", "two") = linear_map::LinearMap(new linear_map::DenseMatrixImpl(A12));
  A("two", "one") = linear_map::LinearMap(
      new linear_map::DenseMatrixImpl(A12.transpose()));
  A("two", "two") = linear_map::Scalar(10, 2);

  Eigen::VectorXd b1 = Eigen::VectorXd::Random(5);
  Eigen::VectorXd b2 = Eigen::VectorXd::Random(2);
  BlockVector b;
  b("one") = b1;
  b("two") = b2;
  BlockCholesky chol;
  chol.Compute(A);
  BlockVector x = chol.Solve(b);

  Eigen::MatrixXd A0 = 10*Eigen::MatrixXd::Identity(7, 7);
  A0.block(0, 5, 5, 2) = A12;
  A0.block(5, 0, 2, 5) = A12.transpose();
  Eigen::LLT<Eigen::MatrixXd> llt;
  llt.compute(A0);
  CHECK_EQ(Eigen::Success, llt.info());
  Eigen::VectorXd x0 = llt.solve(VStack(b1, b2));

  EXPECT_TRUE(VectorEquals(x0.segment(0, 5), x("one"), 1e-8));
  EXPECT_TRUE(VectorEquals(x0.segment(5, 2), x("two"), 1e-8));
}
