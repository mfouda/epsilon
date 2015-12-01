#include "epsilon/prox/elementwise.h"

class NonNegativeProx final : public ElementwiseProx {
protected:
  Eigen::VectorXd ApplyElementwise(
      const Eigen::VectorXd& lambda,
      const Eigen::VectorXd& v) override {
    return v.cwiseMax(0);
  }
};
REGISTER_PROX_OPERATOR(NON_NEGATIVE, NonNegativeProx);
