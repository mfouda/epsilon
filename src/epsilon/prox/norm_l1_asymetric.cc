#include "epsilon/prox/prox.h"
#include "epsilon/prox/scaled_zone.h"

class NormL1AsymetricProx final : public ScaledZoneProx {
public:
  NormL1AsymetricProx() : ScaledZoneProx(0, 0, 0, 0) {};

  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    alpha_ = arg.f_expr().arg(0).arg(0).arg(0).constant().scalar();
    beta_ = arg.f_expr().arg(0).arg(1).arg(0).constant().scalar();
  }
};
REGISTER_PROX_OPERATOR(NormL1AsymetricProx);

class NormL1AsymetricEpigraph final : public ScaledZoneEpigraph {
public:
  NormL1AsymetricEpigraph() : ScaledZoneEpigraph(0, 0, 0, 0) {};

  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    alpha_ = arg.f_expr().arg(1).arg(0).arg(0).arg(0).constant().scalar();
    beta_ = arg.f_expr().arg(1).arg(0).arg(1).arg(0).constant().scalar();
  }
};
REGISTER_PROX_OPERATOR(NormL1AsymetricEpigraph);
