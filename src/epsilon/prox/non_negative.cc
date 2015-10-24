#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/prox/ortho_invariant.h"

// I(alpha*x + b >= 0)
//
// Expression tree:
// INDICATOR (cone: NON_NEGATIVE)
//   SCALAR_MULTIPLE (alpha*x + b >= 0)
class NonNegativeProx final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    affine::BuildScalarAffineOperator(arg.f_expr().arg(0), &alpha_, &b_);
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    return (1/alpha_)*((alpha_*v + b_).eval().cwiseMax(0) - b_);
  }

private:
  double alpha_;
  Eigen::VectorXd b_;
};
REGISTER_PROX_OPERATOR(NonNegativeProx);

class SimpleNonNegativeProx final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {}
  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    return v.cwiseMax(0);
  }
};

class SemidefiniteProx final : public OrthoInvariantProx {
public:
  SemidefiniteProx() : OrthoInvariantProx(std::make_unique<SimpleNonNegativeProx>(), true, true) {}
};
REGISTER_PROX_OPERATOR(SemidefiniteProx);
