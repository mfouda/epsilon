
#include "epsilon/prox/ortho_invariant.h"

class LambdaMaxProx final : public OrthoInvariantProx {
public:
  LambdaMaxProx() : OrthoInvariantProx(ProxFunction::MAX, true) {}
};
REGISTER_PROX_OPERATOR(LAMBDA_MAX, LambdaMaxProx);
