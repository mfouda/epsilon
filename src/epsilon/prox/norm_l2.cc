#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

// lam*||x||_2
class NormL2Prox final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // NORM_P (p: 2)
    //   VARIABLE (x)
    lambda_ = arg.lambda();
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    const double v_norm = v.norm();
    if (v_norm >= lambda_) {
      return (1 - lambda_/v_norm)*v;
    } else {
      return Eigen::VectorXd::Zero(v.rows());
    }
  }

private:
  double lambda_;
};
REGISTER_PROX_OPERATOR(NormL2Prox);

// I(||x||_2 <= t)
class NormL2Epigraph final : public ProxOperator {
  Eigen::VectorXd Apply(const Eigen::VectorXd& sv) override {
    const int n = sv.rows() - 1;
    const double s = sv(0);
    const double v_norm = sv.tail(n).norm();

    if (v_norm <= -s) {
      return Eigen::VectorXd::Zero(sv.rows());
    } else if (v_norm <= s) {
      return sv;
    } else {
      const double alpha = 0.5*(1 + s/v_norm);
      Eigen::VectorXd tx = alpha*sv;
      tx(0) = alpha*v_norm;
      return tx;
    }
  }
};
REGISTER_PROX_OPERATOR(NormL2Epigraph);
