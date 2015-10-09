#include "epsilon/prox/prox.h"
#include "epsilon/prox/scaled_zone.h"
#include "epsilon/prox/ortho_invariant.h"

class NormL1Prox final : public ScaledZoneProx {
public:
  NormL1Prox() : ScaledZoneProx(1., 1., 0., 0.) {};
};
REGISTER_PROX_OPERATOR(NormL1Prox);

class NormL1Epigraph final : public ScaledZoneEpigraph {
public:
  NormL1Epigraph() : ScaledZoneEpigraph(1., 1., 0., 0.) {};
};
REGISTER_PROX_OPERATOR(NormL1Epigraph);

class NormNuclearProx final : public OrthoInvariantProx {
public:
  NormNuclearProx() : OrthoInvariantProx(std::make_unique<NormL1Prox>()) {}
};
REGISTER_PROX_OPERATOR(NormNuclearProx);

class NormNuclearEpigraph final : public OrthoInvariantEpigraph {
public:
  NormNuclearEpigraph() : OrthoInvariantEpigraph(std::make_unique<NormL1Epigraph>()) {}
};
REGISTER_PROX_OPERATOR(NormNuclearEpigraph);
