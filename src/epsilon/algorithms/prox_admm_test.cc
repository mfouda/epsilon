

#include <gtest/gtest.h>
#include <gflags/gflags.h>

#include "distopt/algorithms/algorithm_testutil.h"
#include "distopt/algorithms/consensus_prox.h"
#include "distopt/expression/expression.h"
#include "distopt/expression/problem.h"
#include "distopt/expression/expression_testutil.h"
#include "distopt/parameters/local_parameter_service.h"
#include "distopt/util/problems.h"
#include "distopt/util/string.h"
#include "distopt/util/vector_testutil.h"

class ConsensusProxSolverTest : public testing::Test {
 protected:
  ConsensusProxSolverTest() : var_id_("x") {
    params_.set_max_iterations(100);
  }

  void Solve() {
    ConsensusProxSolver cprox(
        problem_, params_,
        std::unique_ptr<ParameterService>(new LocalParameterService));
    cprox.Solve();
    status_ = cprox.status();

    LocalParameterService parameter_service;
    for (const Expression* expr : GetVariables(problem_)) {
      vars_[expr->variable().variable_id()] =
          parameter_service.Fetch(
              VariableId(cprox.problem_id(), expr->variable().variable_id()));
    }
  }

  void BuildLS(int m, int n, int k, bool consensus,
               Eigen::MatrixXd* A, Eigen::VectorXd* b) {
    srand(0);

    std::vector<Expression> xs;
    for (int i = 0; i < k; i++) {
      Eigen::MatrixXd Ai = Eigen::MatrixXd::Random(m, n);
      Eigen::VectorXd bi = Eigen::VectorXd::Random(m);
      Expression x = expression::Variable(n, 1, "x" + std::to_string(i));

      ProxFunction* f = problem_.add_prox_function();
      f->set_function(ProxFunction::SUM_SQUARES);
      f->set_alpha(1);
      *f->mutable_arg() = expression::Add({
          expression::Multiply(TestConstant(Ai), x),
          expression::Negate(TestConstant(bi))});

      *A = Stack(*A, Ai);
      *b = Stack(*b, bi);
      xs.push_back(x);
    }

    std::vector<Expression> constrs;

    if (consensus) {
      Expression z = expression::Variable(n, 1, "z");
      ConsensusVariable* cv = problem_.add_consensus_variable();
      cv->set_variable_id("z");
      cv->set_num_instances(k);
      for (int i = 0; i < k; i++) {
        constrs.push_back(expression::Add(xs[i], expression::Negate(z)));
      }
    } else {
      for (int i = 0; i < k - 1; i++) {
        constrs.push_back(expression::Add(xs[i], expression::Negate(xs[i+1])));
      }
    }
    *problem_.mutable_equality_constraint() = expression::VStack(constrs);
  }

  void BuildNorm1(int n) {
      ProxFunction* f = problem_.add_prox_function();
      f->set_function(ProxFunction::NORM_1);
      f->set_alpha(1);
      *f->mutable_arg() = TestVariable(n, 1);
  }

  void BuildLasso(int m, int n, double lambda) {
      Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
      Eigen::VectorXd b = Eigen::VectorXd::Random(m);

      Expression x0 = expression::Variable(n, 1, "x0");
      Expression x1 = expression::Variable(n, 1, "x1");

      {
        ProxFunction* f = problem_.add_prox_function();
        f->set_function(ProxFunction::SUM_SQUARES);
        f->set_alpha(1);
        *f->mutable_arg() =
            expression::Add(
                expression::Multiply(TestConstant(A), x0),
                expression::Negate(TestConstant(b)));
      }

      {
        ProxFunction* f = problem_.add_prox_function();
        f->set_function(ProxFunction::NORM_1);
        f->set_alpha(lambda);
        *f->mutable_arg() = x1;
      }

      *problem_.mutable_equality_constraint() = expression::Add(
          x0, expression::Negate(x1));
  }

  void BuildTV(int n, double lambda) {
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);

    Expression x = expression::Variable(n, 1, "x");
    Expression u = expression::Variable(n-1, 1, "u");

    {
      ProxFunction* f = problem_.add_prox_function();
      f->set_function(ProxFunction::SUM_SQUARES);
      f->set_alpha(1);
      *f->mutable_arg() =
          expression::Add(x, expression::Negate(TestConstant(b)));
    }

    {
      ProxFunction* f = problem_.add_prox_function();
      f->set_function(ProxFunction::NORM_1);
      f->set_alpha(lambda);
      *f->mutable_arg() = u;
    }

    *problem_.mutable_equality_constraint() = expression::Add(
        expression::Add(
            expression::Index(0, n-1, x),
            expression::Negate(
                expression::Index(1, n-1, x))),
        expression::Negate(u));
  }

  std::string var_id_;
  ProxProblem problem_;
  SolverParams params_;
  ProblemStatus status_;

  std::unordered_map<std::string, Eigen::VectorXd, std::hash<std::string>,
                     std::equal_to<std::string>,
                     Eigen::aligned_allocator<VectorXd>> vars_;

};

TEST_F(ConsensusProxSolverTest, LS_Split2) {
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  BuildLS(10, 5, 2, false, &A, &b);
  Solve();
  EXPECT_EQ(status_.state(), ProblemStatus::OPTIMAL);
  EXPECT_TRUE(VectorEquals(ComputeLS(A, b), vars_["x0"], 1e-2));
  EXPECT_TRUE(VectorEquals(ComputeLS(A, b), vars_["x1"], 1e-2));
}

TEST_F(ConsensusProxSolverTest, LS_Split2_Consensus) {
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  BuildLS(3, 2, 2, true, &A, &b);
  Solve();
  EXPECT_EQ(status_.state(), ProblemStatus::OPTIMAL);
  EXPECT_TRUE(VectorEquals(ComputeLS(A, b), vars_["x0"], 1e-2));
  EXPECT_TRUE(VectorEquals(ComputeLS(A, b), vars_["x1"], 1e-2));
  EXPECT_TRUE(VectorEquals(ComputeLS(A, b), vars_["z"], 1e-2));
}

TEST_F(ConsensusProxSolverTest, Lasso) {
  BuildLasso(5, 10, 0.1);
  Solve();
  EXPECT_EQ(status_.state(), ProblemStatus::OPTIMAL);
  // TODO(mwytock): Verify result optimality
}

TEST_F(ConsensusProxSolverTest, TV) {
  BuildTV(10, 1);
  Solve();
  EXPECT_EQ(status_.state(), ProblemStatus::OPTIMAL);
  // TODO(mwytock): Verify result optimality
}
