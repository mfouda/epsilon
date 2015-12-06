#include "epsilon/prox/vector_prox.h"

class NonNegativeProx final : public VectorProx {
protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override {
    output->set_value(0, input.value_vec(0).cwiseMax(0));
  }
};
REGISTER_PROX_OPERATOR(NON_NEGATIVE, NonNegativeProx);
