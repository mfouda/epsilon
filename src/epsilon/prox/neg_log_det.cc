
#include "epsilon/prox/ortho_invariant.h"

class NegLogDetProx final : public OrthoInvariantProx {
public:
  NegLogDetProx() : OrthoInvariantProx(ProxFunction::SUM_NEG_LOG, true) {}
};
REGISTER_PROX_OPERATOR(NEG_LOG_DET, NegLogDetProx);
