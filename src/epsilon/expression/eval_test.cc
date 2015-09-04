
#include <gtest/gtest.h>

#include "distopt/expression/eval.h"
#include "distopt/expression/expression.h"
#include "distopt/expression/expression_testutil.h"
#include "distopt/expression/problem.h"
#include "distopt/parameters/local_parameter_service.h"
#include "distopt/util/string.h"
#include "distopt/util/vector.h"

class EvalTest : public testing::Test {
 protected:
  EvalTest() : evaluator_(
      std::unique_ptr<ParameterService>(new LocalParameterService())) {}

  void Set(
      const std::string& variable_id, const Eigen::MatrixXd& value) {
    parameter_service_.Update(
        VariableId(problem_id(), variable_id),
        ToVector(value));
  }

  double Evaluate(const Expression& expr) {
    Eigen::MatrixXd V = evaluator_.Evaluate(problem_id(), expr);
    CHECK_EQ(V.size(), 1);
    return V(0, 0);
  }

  uint64_t problem_id() {
    const testing::TestInfo* test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    return std::hash<std::string>()(
        StringPrintf("%s_%s", test_info->name(), test_info->test_case_name()));
  }

  LocalParameterService parameter_service_;
  ExpressionEvaluator evaluator_;
};

TEST_F(EvalTest, Covsel) {
  const int n = 10;
  const double lambda = 0.1;

  srand(0);
  Eigen::MatrixXd S = Eigen::MatrixXd::Random(n, n);

  Expression X = expression::Variable(n, n, "X");
  Expression expr = expression::Add(
      expression::Negate(expression::LogDet(X)),
      expression::Sum(expression::MultiplyElementwise(TestConstant(S), X)),
      expression::Multiply(
          expression::ScalarConstant(lambda),
          expression::PNorm(X, 1)));

  Eigen::MatrixXd X0 = Eigen::MatrixXd::Random(n, n);
  X0 = X0.transpose()*X0;
  Set("X", X0);

  const double obj_val = -log(X0.determinant()) +
                         (S.array()*X0.array()).sum() +
                         lambda*X0.array().abs().sum();
  EXPECT_NEAR(Evaluate(expr), obj_val, 1e-8);
}

TEST_F(EvalTest, Lasso) {
  const int m = 5;
  const int n = 10;
  const double lambda = 0.1;

  srand(0);
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
  Eigen::VectorXd b = Eigen::VectorXd::Random(m);

  Expression x = expression::Variable(n, 1, "x");
  Expression expr = expression::Add(
      expression::Power(
          expression::PNorm(
              expression::Add(
                  expression::Multiply(TestConstant(A), x),
                  expression::Negate(TestConstant(b))), 2), 2),
      expression::Multiply(
          expression::ScalarConstant(lambda),
          expression::PNorm(x, 1)));

  Eigen::VectorXd x0 = Eigen::VectorXd::Random(n);
  Set("x", x0);

  const double obj_val = (A*x0 - b).array().square().sum() +
                         lambda*x0.array().abs().sum();
  EXPECT_NEAR(Evaluate(expr), obj_val, 1e-8);
}

TEST_F(EvalTest, TVColorImage) {
  const int n = 10;
  const int k = 3;
  const double lambda = 0.1;

  srand(0);
  std::vector<Eigen::MatrixXd> Y(k);
  std::vector<Expression> X(k);
  std::vector<Expression> norm_terms;
  std::vector<Expression> obj_terms;

  for (int i = 0; i < k; i++) {
    Y[i] = Eigen::MatrixXd::Random(n, n);
    X[i] = expression::Variable(n, n, "X" + std::to_string(i), n*n*i);

    obj_terms.push_back(expression::Power(
        expression::PNorm(
            expression::Add(
                X[i],
                expression::Negate(TestConstant(Y[i]))), 2), 2));

    norm_terms.push_back(expression::Add(
        expression::Index(0, n-1, 1, n-1, X[i]),
        expression::Negate(
            expression::Index(0, n-1, 0, n-1, X[i]))));

    norm_terms.push_back(expression::Add(
        expression::Index(1, n-1, 0, n-1, X[i]),
        expression::Negate(
            expression::Index(0, n-1, 0, n-1, X[i]))));
  }

  obj_terms.push_back(expression::Multiply(
      expression::ScalarConstant(lambda),
      expression::Sum(
          expression::Norm2Elementwise(norm_terms))));
  Expression expr = expression::Add(obj_terms);

  // Compute objective directly
  double obj_val = 0;
  Eigen::MatrixXd DX_norm_squared = Eigen::MatrixXd::Zero(n-1, n-1);
  for (int i = 0; i < k; i++) {
    Eigen::MatrixXd Xi = Eigen::MatrixXd::Random(n, n);
    Set("X" + std::to_string(i), Xi);

    obj_val += (Xi - Y[i]).array().square().sum();
    DX_norm_squared.array() += (
        Xi.block(0, 0, n-1, n-1) -
        Xi.block(0, 1, n-1, n-1)).array().square();
    DX_norm_squared.array() += (
        Xi.block(0, 0, n-1, n-1) -
        Xi.block(1, 0, n-1, n-1)).array().square();
  }
  obj_val += lambda*DX_norm_squared.array().sqrt().sum();
  EXPECT_NEAR(Evaluate(expr), obj_val, 1e-8);
}
