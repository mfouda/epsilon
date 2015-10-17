

#include <gtest/gtest.h>

#include "epsilon/algorithms/algorithm_testutil.h"
#include "epsilon/algorithms/prox_admm.h"
#include "epsilon/expression/expression.h"
#include "epsilon/expression/expression_testutil.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/parameters/local_parameter_service.h"
#include "epsilon/util/string.h"
#include "epsilon/vector/vector_testutil.h"

class ProxADMMSolverTest : public testing::Test {
 protected:
  ProxADMMSolverTest() : var_id_("x") {
    params_.set_max_iterations(100);
  }

  void Solve() {
    ProxADMMSolver cprox(
        problem_, params_,
        std::unique_ptr<ParameterService>(new LocalParameterService));
    cprox.Solve();
    status_ = cprox.status();

    for (int i = 0; i < cprox.N_; i++)
      x_ += cprox.x_[i];
  }

  void BuildLS(int m, int n, int k, Eigen::MatrixXd* A, Eigen::VectorXd* b) {
    srand(0);

    std::vector<Expression> fs;
    std::vector<Expression> xs;

    for (int i = 0; i < k; i++) {
      Eigen::MatrixXd Ai = Eigen::MatrixXd::Random(m, n);
      Eigen::VectorXd bi = Eigen::VectorXd::Random(m);

      Expression x = expression::Variable(n, 1, "x" + std::to_string(i));
      Expression f = expression::Power(
          expression::NormP(
              expression::Add(
                  expression::Multiply(TestConstant(Ai), x),
                  expression::Negate(TestConstant(bi))), 2), 2);
      f.mutable_proximal_operator()->set_name("LeastSquaresProx");

      *A = VStack(*A, Ai);
      *b = VStack(*b, bi);
      xs.push_back(x);
      fs.push_back(f);
    }

    for (int i = 0; i < k - 1; i++) {
      *problem_.add_constraint() =
          expression::Indicator(
              Cone::ZERO,
              expression::Add(xs[i], expression::Negate(xs[i+1])));
    }

    *problem_.mutable_objective() = expression::Add(fs);
  }

  void BuildLasso(int m, int n, double lambda) {
    srand(0);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
    Eigen::VectorXd b = Eigen::VectorXd::Random(m);

    Expression x0 = expression::Variable(n, 1, "x0");
    Expression x1 = expression::Variable(n, 1, "x1");

    Expression f =
        expression::Power(
            expression::NormP(
                expression::Add(
                    expression::Multiply(TestConstant(A), x0),
                    expression::Negate(TestConstant(b))), 2), 2);
    f.mutable_proximal_operator()->set_name("LeastSquaresProx");

    Expression g = expression::NormP(x1, 1);
    g.mutable_proximal_operator()->set_name("NormL1Prox");

    *problem_.mutable_objective() = expression::Add(
        f, expression::Multiply(expression::Constant(lambda), g));
    *problem_.add_constraint() = expression::Indicator(
        Cone::ZERO,
        expression::Add(x0, expression::Negate(x1)));
  }

  void BuildTV(int n, double lambda) {
    srand(0);
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);

    Expression x = expression::Variable(n, 1, "x");
    Expression u = expression::Variable(n-1, 1, "u");

    Expression f = expression::Power(
        expression::NormP(
                expression::Add(
                    x,
                    expression::Negate(TestConstant(b))), 2), 2);
    f.mutable_proximal_operator()->set_name("LeastSquaresProx");

    Expression g = expression::NormP(u, 1);
    g.mutable_proximal_operator()->set_name("NormL1Prox");

    *problem_.mutable_objective() = expression::Add(
        f, expression::Multiply(expression::Constant(lambda), g));
    *problem_.add_constraint() = expression::Indicator(
        Cone::ZERO,
        expression::Add(
            expression::Add(
                expression::Index(0, n-1, x),
                expression::Negate(
                    expression::Index(1, n-1, x))),
            expression::Negate(u)));
  }

  void BuildLeastAbsDeviations(int m, int n) {
    srand(0);

    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
    Eigen::VectorXd b = Eigen::VectorXd::Random(m);

    Expression y = expression::Variable(m, 1, "y");
    Expression x = expression::Variable(n, 1, "x");

    Expression f = expression::NormP(y, 1);
    f.mutable_proximal_operator()->set_name("NormL1Prox");

    *problem_.mutable_objective() = expression::Add({f});
    *problem_.add_constraint() = expression::Indicator(
        Cone::ZERO,
        expression::Add(
            expression::Add(
                expression::Multiply(TestConstant(A), x),
                expression::Negate(TestConstant(b))),
            expression::Negate(y)));
  }

  std::string var_id_;
  Problem problem_;
  SolverParams params_;
  SolverStatus status_;
  BlockVector x_;
};

TEST_F(ProxADMMSolverTest, LS_Split2) {
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  BuildLS(10, 5, 2, &A, &b);
  Solve();
  EXPECT_EQ(status_.state(), SolverStatus::OPTIMAL);
  EXPECT_TRUE(VectorEquals(ComputeLS(A, b), x_("x0"), 1e-2));
  EXPECT_TRUE(VectorEquals(ComputeLS(A, b), x_("x1"), 1e-2));
}

TEST_F(ProxADMMSolverTest, Lasso) {
  BuildLasso(5, 10, 0.1);
  Solve();
  EXPECT_EQ(status_.state(), SolverStatus::OPTIMAL);
  // TODO(mwytock): Verify result optimality
}

// TODO(mwytock): Fix this
// TEST_F(ProxADMMSolverTest, TV) {
//   BuildTV(10, 1);
//   Solve();
//   EXPECT_EQ(status_.state(), SolverStatus::OPTIMAL);
//   // TODO(mwytock): Verify result optimality
// }

TEST_F(ProxADMMSolverTest, LeastAbsDeviations) {
  BuildLeastAbsDeviations(3, 2);
  params_.set_max_iterations(100);
  Solve();
  EXPECT_EQ(status_.state(), SolverStatus::OPTIMAL);
  // TODO(mwytock): Verify result optimality
}
