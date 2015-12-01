#include "epsilon/prox/vector.h"
#include "epsilon/vector/vector_util.h"

// lam*||x||_2
class Norm2Prox final : public VectorProx {
  Eigen::VectorXd ApplyVector(
      double lambda,
      const Eigen::VectorXd& v) override {
    const double v_norm = v.norm();
    if (v_norm >= lambda) {
      return (1 - lambda/v_norm)*v;
    } else {
      return Eigen::VectorXd::Zero(v.rows());
    }
  }
};
REGISTER_PROX_OPERATOR(NORM_2, Norm2Prox);
