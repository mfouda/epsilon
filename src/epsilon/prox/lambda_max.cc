
#include "epsilon/prox/ortho_invariant.h"
#include "epsilon/prox/max.h"

class LambdaMaxProx final : public OrthoInvariantProx {
public:
  LambdaMaxProx() : OrthoInvariantProx(std::make_unique<MaxProx>(), true) {}
};
REGISTER_PROX_OPERATOR(LAMBDA_MAX, LambdaMaxProx);
