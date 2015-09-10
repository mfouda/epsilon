
#include <gtest/gtest.h>

#include "epsilon/operators/prox.h"
#include "epsilon/expression/expression_testutil.h"
#include "epsilon/expression/expression.h"
#include "epsilon/util/vector_testutil.h"


void TestNormL1Weighted(
    const std::vector<double>& v,
    const std::vector<double>& expected_x,
    const std::vector<double>& weights,
    double lambda) {
  const int n = v.size();
  ProxFunction f;
  f.set_alpha(1);
  f.set_function(ProxFunction::NORM_1);
  *f.add_arg() = expression::MultiplyElementwise(
      TestConstant(Eigen::Map<const Eigen::VectorXd>(&weights[0], n)),
      TestVariable(n, 1));

  std::unique_ptr<VectorOperator> prox(CreateProxOperator(f, lambda, n));
  prox->Init();
  Eigen::VectorXd x = prox->Apply(Eigen::Map<const Eigen::VectorXd>(&v[0], n));

  EXPECT_TRUE(VectorEquals(
      Eigen::Map<const Eigen::VectorXd>(&expected_x[0], n), x, 1e-8));
}

void TestNormL1(const std::vector<double>& v,
                const std::vector<double>& expected_x,
                double lambda) {
  std::vector<double> weights;
  for (int i = 0; i < v.size(); i++) {
    weights.push_back(1);
  }
  TestNormL1Weighted(v, expected_x, weights, lambda);
}

void TestNormL1L2(const std::vector<double>& v,
                  const std::vector<double>& expected_x,
                  double lambda, int m, int n) {

  ProxFunction f;
  f.set_function(ProxFunction::NORM_1_2);
  f.set_alpha(1);
  *f.add_arg() = expression::Variable(m, n, "X");

  std::unique_ptr<VectorOperator> prox(CreateProxOperator(f, lambda, m*n));
  prox->Init();
  Eigen::VectorXd x = prox->Apply(
      Eigen::Map<const Eigen::VectorXd>(&v[0], m*n));

  EXPECT_TRUE(VectorEquals(
      Eigen::Map<const Eigen::VectorXd>(&expected_x[0], m*n), x, 1e-2));
}

void TestSumSquares(int m, int n, double lambda) {
  srand(0);
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
  Eigen::VectorXd b = Eigen::VectorXd::Random(m);

  ProxFunction f;
  f.set_function(ProxFunction::SUM_SQUARES);
  f.set_alpha(1);
  *f.add_arg() =
      expression::Add({
          expression::Multiply(
              TestConstant(A),
              TestVariable(n, 1)),
          expression::Negate(TestConstant(b))});

  std::unique_ptr<VectorOperator> prox(CreateProxOperator(f, lambda, n));
  prox->Init();
  Eigen::VectorXd v = Eigen::VectorXd::Random(n);
  Eigen::VectorXd x = prox->Apply(v);

  Eigen::LLT<Eigen::MatrixXd> solver;
  solver.compute(Eigen::MatrixXd::Identity(n, n) +
                 2*lambda*A.transpose()*A);
  ASSERT_EQ(solver.info(), Eigen::Success);

  Eigen::VectorXd x_expected = solver.solve(2*lambda*A.transpose()*b + v);
  ASSERT_EQ(solver.info(), Eigen::Success);

  EXPECT_TRUE(VectorEquals(x, x_expected, 1e-8));
}

void TestNegativeLogDet(int n, double lambda) {
  ProxFunction f;
  f.set_function(ProxFunction::NEGATIVE_LOG_DET);
  f.set_alpha(1);
  *f.add_arg() = TestVariable(n, n);

  srand(0);
  Eigen::MatrixXd V = Eigen::MatrixXd::Random(n, n);
  V = (V + V.transpose()).eval();

  std::unique_ptr<VectorOperator> prox(CreateProxOperator(f, lambda, n*n));
  prox->Init();
  Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(
      prox->Apply(
          Eigen::Map<const Eigen::VectorXd>(V.data(), n*n)).data(),
      n, n);

  // Check optimality condition
  EXPECT_TRUE(MatrixEquals(X - lambda*X.inverse(), V, 1e-8));
}

// Helper function
Eigen::VectorXd TestProx(
    const ProxFunction::Function& function,
    const Expression& arg,
    double lambda,
    const Eigen::VectorXd& v) {
  const int n = GetVariableDimension(arg);

  ProxFunction f;
  f.set_function(function);
  f.set_alpha(1);
  *f.add_arg() = arg;

  std::unique_ptr<VectorOperator> prox(CreateProxOperator(f, lambda, n));
  prox->Init();
  return prox->Apply(v);
}

TEST(SumSquares, FatA) {
  const int m = 5;
  const int n = 10;
  TestSumSquares(m, n, 1);
  TestSumSquares(m, n, 0.1);
}

TEST(SumSquares, SkinnyA) {
  const int m = 10;
  const int n = 5;
  TestSumSquares(m, n, 1);
  TestSumSquares(m, n, 0.1);
}

TEST(NormL1, Basic) {
  TestNormL1({1, 2, 3}, {0, 1, 2}, 1);
  TestNormL1({-1, -2, 3}, {0, -1, 2}, 1);
  TestNormL1({-1, -2, 3}, {0, 0, 0.5}, 2.5);
}

TEST(NormL1, Weighted) {
  TestNormL1Weighted({1, 2, 3}, {1, 0, 0}, {0, 2, 3}, 1);
  TestNormL1Weighted({-1, -2, 3}, {-1, 0, 0}, {0, 2, 3}, 1);
  TestNormL1Weighted({-1, -2, 3}, {-1, -1, 1.5}, {0, 2, 3}, 0.5);
}

TEST(NegativeLogDet, Basic) {
  TestNegativeLogDet(10, 1);
  TestNegativeLogDet(10, 2);
}

TEST(NormL1L2, Basic) {
  const int m = 2;
  const int n = 3;

  // Column order, so first example is [1 3 4; 2 5 6]
  TestNormL1L2({1, 2, 3, 4, 5, 6},
               {0.83, 1.73, 2.49, 3.46, 4.15, 5.19},
               1, m, n);
  TestNormL1L2({1, 2, 3, 4, 5, 6},
               {0, 0.39, 0, 0.79, 0, 1.19},
               6, m, n);
}
