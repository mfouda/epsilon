#include "epsilon/prox/prox.h"
#include "epsilon/prox/scaled_zone.h"

class HingeProx final : public ScaledZoneProx {
public:
  HingeProx() : ScaledZoneProx(0., 1., 1., 0.) {};
};
REGISTER_PROX_OPERATOR(HingeProx);

class HingeEpigraph final : public ScaledZoneEpigraph {
public:
  HingeEpigraph() : ScaledZoneEpigraph(0., 1., 1., 0.) {};
};
REGISTER_PROX_OPERATOR(HingeEpigraph);
