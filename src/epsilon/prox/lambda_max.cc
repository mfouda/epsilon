
#include "epsilon/prox/ortho_invariant.h"

class LambdaMaxProx final : public OrthoInvariantProx {
public:
  LambdaMaxProx() : OrthoInvariantProx(ProxFunction::MAX, true) {}
};
REGISTER_PROX_OPERATOR(LAMBDA_MAX, LambdaMaxProx);

class LambdaMaxEpigraph final : public OrthoInvariantProx {
public:
  LambdaMaxEpigraph() : OrthoInvariantProx(
      ProxFunction::MAX, true, false, true) {}
};
REGISTER_EPIGRAPH_OPERATOR(LAMBDA_MAX, LambdaMaxEpigraph);
