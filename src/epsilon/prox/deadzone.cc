#include "epsilon/prox/scaled_zone.h"

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

class DeadZoneProx final : public ScaledZoneProx {
public:
  DeadZoneProx() : ScaledZoneProx(1, 1, 0, 0) {};

  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    M_ = arg.f_expr().arg(0).arg(0).arg(0).arg(1).arg(0).constant().scalar();
    VLOG(1) << "Deadzone: M = " << M_ << "\n";
  }
};
REGISTER_PROX_OPERATOR(DeadZoneProx);
