#include "epsilon/prox/prox.h"
#include "epsilon/prox/scaled_zone.h"

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
