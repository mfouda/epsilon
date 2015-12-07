#include "epsilon/prox/vector_prox.h"

// lam*||x||_2
class Norm2Prox final : public VectorProx {
protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override {
    const double lambda = input.lambda();
    const Eigen::VectorXd& v = input.value_vec(0);
    const double v_norm = v.norm();
    if (v_norm >= lambda) {
      output->set_value(0, (1 - lambda/v_norm)*v);
    } else {
      output->set_value(0, Eigen::VectorXd::Zero(v.rows()));
    }
  }
};
REGISTER_PROX_OPERATOR(NORM_2, Norm2Prox);
