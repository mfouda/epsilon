
#include "epsilon/prox/ortho_invariant.h"

class NegLogDetProx final : public OrthoInvariantProx {
public:
  NegLogDetProx() : OrthoInvariantProx(ProxFunction::SUM_NEG_LOG, true) {}
};
REGISTER_PROX_OPERATOR(NEG_LOG_DET, NegLogDetProx);

class NegLogDetEpigraph final : public OrthoInvariantProx {
public:
  NegLogDetEpigraph() : OrthoInvariantProx(
      ProxFunction::SUM_NEG_LOG, true, false, true) {}
};
REGISTER_EPIGRAPH_OPERATOR(NEG_LOG_DET, NegLogDetEpigraph);
