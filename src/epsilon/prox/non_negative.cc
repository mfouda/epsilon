#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/elementwise.h"
//#include "epsilon/prox/ortho_invariant.h"

// I(x >= 0)
class NonNegativeProx final : public ElementwiseProxOperator {
public:
  Eigen::VectorXd ApplyElementwise(const Eigen::VectorXd& v) override {
    return v.cwiseMax(0);
  }
};
REGISTER_PROX_OPERATOR(NON_NEGATIVE, NonNegativeProx);

// class SimpleNonNegativeProx final : public ProxOperator {
// public:
//   void Init(const ProxOperatorArg& arg) override {}
//   Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
//     return v.cwiseMax(0);
//   }
// };

// class SemidefiniteProx final : public OrthoInvariantProx {
// public:
//   SemidefiniteProx() : OrthoInvariantProx(std::make_unique<SimpleNonNegativeProx>(), true, true) {}
// };
// REGISTER_PROX_OPERATOR(SemidefiniteProx);
