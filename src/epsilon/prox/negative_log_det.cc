#include <glog/logging.h>

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

// -lam*log|X|
class NegativeLogDetProx final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // NEGATE
    //   LOG_DET
    //     VARIABLE (X)
    const Expression& var_expr = arg.f_expr().arg(0).arg(0);
    CHECK_EQ(var_expr.expression_type(), Expression::VARIABLE);
    CHECK_EQ(GetDimension(var_expr, 0),
             GetDimension(var_expr, 1));
    n_ = GetDimension(var_expr, 0);
    lambda_ = arg.lambda();
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    Eigen::MatrixXd V = ToMatrix(v, n_, n_);
    V = (V + V.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(V);
    CHECK_EQ(solver.info(), Eigen::Success);
    const Eigen::VectorXd& d = solver.eigenvalues();
    const Eigen::MatrixXd& U = solver.eigenvectors();

    Eigen::VectorXd x_tilde =
        (d.array() + (d.array().square() + 4*lambda_).sqrt())/2;
    Eigen::MatrixXd X = U*x_tilde.asDiagonal()*U.transpose();

    return ToVector(X);
  }

private:
  double lambda_;
  int n_;
};
REGISTER_PROX_OPERATOR(NegativeLogDetProx);
