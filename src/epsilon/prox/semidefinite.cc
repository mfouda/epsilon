#include "epsilon/prox/ortho_invariant.h"

class SemidefiniteProx final : public OrthoInvariantProx {
public:
  SemidefiniteProx()
      : OrthoInvariantProx(ProxFunction::NON_NEGATIVE, true, true) {}
};
REGISTER_PROX_OPERATOR(SEMIDEFINITE, SemidefiniteProx);
