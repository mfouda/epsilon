#include "epsilon/prox/ortho_invariant.h"

class NormNuclearProx final : public OrthoInvariantProx {
public:
  NormNuclearProx() : OrthoInvariantProx(ProxFunction::NORM_1) {}
};
REGISTER_PROX_OPERATOR(NORM_NUCLEAR, NormNuclearProx);
