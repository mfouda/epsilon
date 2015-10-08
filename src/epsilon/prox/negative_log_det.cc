#include <glog/logging.h>

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

// -lam*log|X|
class NegativeLogDetProx : public ProxOperator {
public:
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

protected:
  double lambda_;
  int n_;
};
REGISTER_PROX_OPERATOR(NegativeLogDetProx);

// [-log|X| <= t]
class NegativeLogDetEpigraph : public NegativeLogDetProx {
public:
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    const Expression& var_expr = arg.f_expr().arg(0).arg(1).arg(0).arg(0).arg(0);
    CHECK_EQ(var_expr.expression_type(), Expression::VARIABLE);
    CHECK_EQ(GetDimension(var_expr, 0),
             GetDimension(var_expr, 1));
    n_ = GetDimension(var_expr, 0);
    lambda_ = 0;
  }
  Eigen::VectorXd Apply(const Eigen::VectorXd& sv) override {
    lambda_ = 1;

    double s = sv(0);
    Eigen::MatrixXd V = ToMatrix(sv.tail(n_*n_), n_, n_);
    V = (V+V.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(V);
    CHECK_EQ(solver.info(), Eigen::Success);
    const Eigen::VectorXd& d = solver.eigenvalues();
    //VLOG(1) << "\neigen values = " << VectorDebugString(d) << "\n";

    const double eps = 1e-10;

    int iter = 0;
    for(; iter<100; iter++) {
      double g = n_*std::log(2) - lambda_ - s;
      for(int i=0; i<n_; i++)
        g -= std::log(d(i) + std::sqrt(d(i)*d(i)+4*lambda_));

      if(std::fabs(g) <= eps)
        break;

      double h = -1;
      for(int i=0; i<n_; i++)
        h -= 2./(d(i)*d(i) + 4*lambda_ + d(i)*std::sqrt(d(i)*d(i)+4*lambda_));

      if(h >= -1e-10)
        h = -1e-10;

      lambda_ -= g/h;
      if(lambda_ <= 1e-10)
        lambda_ = 1e-10;
    }

    //VLOG(1) << "lambda_ = " << lambda_ << "\n";

    VectorXd tx(1+n_*n_);
    tx(0) = s + lambda_;
    tx.tail(n_*n_) = NegativeLogDetProx::Apply(sv.tail(n_*n_));
    //VLOG(1) << VectorDebugString(tx) << "\n";

    return tx;
  }
};
REGISTER_PROX_OPERATOR(NegativeLogDetEpigraph);
