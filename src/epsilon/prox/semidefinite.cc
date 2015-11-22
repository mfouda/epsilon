#include "epsilon/prox/ortho_invariant.h"
#include "epsilon/prox/non_negative.h"

class SemidefiniteProx final : public OrthoInvariantProx {
public:
  SemidefiniteProx()
      : OrthoInvariantProx(std::make_unique<NonNegativeProx>(), true, true) {}
};
REGISTER_PROX_OPERATOR(SEMIDEFINITE, SemidefiniteProx);
