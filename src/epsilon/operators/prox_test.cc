
#include <gtest/gtest.h>

#include "epsilon/expression/expression.h"
#include "epsilon/expression/expression_testutil.h"
#include "epsilon/operators/prox.h"
#include "epsilon/util/vector.h"
#include "epsilon/util/vector_testutil.h"

class ProxOperatorTest : public testing::Test {
 protected:
  ProxOperatorTest() {
    srand(0);
    lambda_ = 1;
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) {
    var_map_.Insert(f_expr_);
    std::unique_ptr<VectorOperator> op = CreateProxOperator(
        lambda_, f_expr_, var_map_);
    op->Init();
    return op->Apply(v);
  }

  double lambda_;
  Expression f_expr_;
  VariableOffsetMap var_map_;
};

class LeastSquaresTest : public ProxOperatorTest {
 protected:
  void GenerateData(int m, int n) {
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

class NormL1Test : public ProxOperatorTest {
 protected:
  void CreateExpression(int n) {
    f_expr_ = expression::NormP(expression::Variable(n, 1, "x"), 1);
  }

  void CreateWeightedExpression(const std::vector<double>& weights) {
    const int n = weights.size();
    f_expr_ = expression::NormP(
        expression::MultiplyElementwise(
            TestConstant(Eigen::Map<const Eigen::VectorXd>(&weights[0], n)),
            expression::Variable(n, 1, "x")), 1);
  }
};

class NormL1L2Test : public ProxOperatorTest {
 protected:
  void CreateExpression(int m, int n) {
    f_expr_ = expression::NormPQ(expression::Variable(m, n, "X"), 1, 2);
  }
};

class NegativeLogDetTest : public ProxOperatorTest {
 protected:
  void CreateExpression(int n) {
    f_expr_ = expression::Negate(
        expression::LogDet(
            expression::Variable(n, n, "X")));
  }
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

TEST_F(NormL1Test, Basic) {
  CreateExpression(3);

  lambda_ = 1;
  EXPECT_TRUE(VectorEquals(Apply(TestVector({1,2,3})), TestVector({0,1,2})));
  EXPECT_TRUE(VectorEquals(Apply(TestVector({-1,-2,3})), TestVector({0,-1,2})));

  lambda_ = 2.5;
  EXPECT_TRUE(VectorEquals(
      Apply(TestVector({-1,-2,3})), TestVector({0,0,0.5})));
}

TEST_F(NormL1Test, Weighted) {
  CreateWeightedExpression({0,2,3});

  lambda_ = 1;
  EXPECT_TRUE(VectorEquals(Apply(TestVector({1,2,3})), TestVector({1,0,0})));
  EXPECT_TRUE(VectorEquals(Apply(TestVector({-1,-2,3})), TestVector({-1,0,0})));

  lambda_ = 0.5;
  EXPECT_TRUE(VectorEquals(
      Apply(TestVector({-1,-2,3})), TestVector({-1,-1,1.5})));
}

TEST_F(NegativeLogDetTest, Basic) {
  const int n = 10;
  CreateExpression(n);

  Eigen::MatrixXd V = Eigen::MatrixXd::Random(n, n);
  V = (V + V.transpose()).eval();

  {
    lambda_ = 1;
    Eigen::MatrixXd X = ToMatrix(Apply(ToVector(V)), n, n);
    EXPECT_TRUE(MatrixEquals(X - lambda_*X.inverse(), V, 1e-8));
  }

  {
    lambda_ = 2;
    Eigen::MatrixXd X = ToMatrix(Apply(ToVector(V)), n, n);
    EXPECT_TRUE(MatrixEquals(X - lambda_*X.inverse(), V, 1e-8));
  }
}

TEST_F(NormL1L2Test, Basic) {
  const int m = 2;
  const int n = 3;
  CreateExpression(m, n);

  // Column order, so first example is [1 3 4; 2 5 6]
  lambda_ = 1;
  EXPECT_TRUE(VectorEquals(
      Apply(TestVector({1, 2, 3, 4, 5, 6})),
      TestVector({0.83, 1.73, 2.49, 3.46, 4.15, 5.19}),
      1e-2));

  lambda_ = 6;
  EXPECT_TRUE(VectorEquals(
      Apply(TestVector({1, 2, 3, 4, 5, 6})),
      TestVector({0, 0.39, 0, 0.79, 0, 1.19}),
      1e-2));
}
