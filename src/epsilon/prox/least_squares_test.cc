
#include <gtest/gtest.h>

#include "epsilon/expression/expression.h"
#include "epsilon/expression/expression_testutil.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"
#include "epsilon/vector/vector_testutil.h"
#include "epsilon/vector/block_matrix.h"

class ProxOperatorTest : public testing::Test {
 protected:
  ProxOperatorTest() {
    srand(0);
    lambda_ = 1;
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v0) {
    v_("0") = v0;
    A_("0", "x") = LinearMap::Identity(v0.rows());

    std::unique_ptr<BlockVectorOperator> op = CreateProxOperator(
        lambda_, A_, f_expr_);
    op->Init();
    return op->Apply(v_)("x");
  }

  double lambda_;
  Expression f_expr_;
  BlockMatrix A_;
  BlockVector v_;
};

class LeastSquaresTest : public ProxOperatorTest {
 protected:
  void GenerateData(int m, int n) {
    srand(0);
    A_ = Eigen::MatrixXd::Random(m, n);
    b_ = Eigen::VectorXd::Random(m);
  }

  void CreateExpression(int n) {
    f_expr_ = expression::Power(
        expression::NormP(
            expression::Add(
                expression::Multiply(
                    TestConstant(A_),
                    expression::Variable(n, 1, "x")),
                expression::Negate(TestConstant(b_))), 2), 2);
    f_expr_.mutable_proximal_operator()->set_name("LeastSquaresProx");
  }

  Eigen::VectorXd ComputeLS(const Eigen::VectorXd& v) {
    const int n = A_.cols();
    Eigen::LLT<Eigen::MatrixXd> solver;
    solver.compute(Eigen::MatrixXd::Identity(n, n) +
                   2*lambda_*A_.transpose()*A_);
    CHECK_EQ(solver.info(), Eigen::Success);

    return solver.solve(2*lambda_*A_.transpose()*b_ + v);
  }

  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
};

TEST_F(LeastSquaresTest, FatA) {
  const int m = 5;
  const int n = 10;

  GenerateData(m, n);
  CreateExpression(n);
  Eigen::VectorXd v = Eigen::VectorXd::Random(n);

  lambda_ = 1;
  EXPECT_TRUE(VectorEquals(Apply(v), ComputeLS(v), 1e-8));

  lambda_ = 0.1;
  EXPECT_TRUE(VectorEquals(Apply(v), ComputeLS(v), 1e-8));
}

TEST_F(LeastSquaresTest, SkinnyA) {
  const int m = 10;
  const int n = 5;

  GenerateData(m, n);
  CreateExpression(n);
  Eigen::VectorXd v = Eigen::VectorXd::Random(n);

  lambda_ = 1;
  EXPECT_TRUE(VectorEquals(Apply(v), ComputeLS(v), 1e-8));

  lambda_ = 0.1;
  EXPECT_TRUE(VectorEquals(Apply(v), ComputeLS(v), 1e-8));
}
