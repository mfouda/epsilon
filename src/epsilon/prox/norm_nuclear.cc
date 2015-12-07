#include "epsilon/prox/ortho_invariant.h"

class NormNuclearProx final : public OrthoInvariantProx {
public:
  NormNuclearProx() : OrthoInvariantProx(ProxFunction::NORM_1) {}
};
REGISTER_PROX_OPERATOR(NORM_NUCLEAR, NormNuclearProx);

class NormNuclearEpigraph final : public OrthoInvariantProx {
public:
  NormNuclearEpigraph() : OrthoInvariantProx(
      ProxFunction::NORM_1, false, false, true) {}
};
REGISTER_EPIGRAPH_OPERATOR(NORM_NUCLEAR, NormNuclearEpigraph);
