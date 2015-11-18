#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
//#include "epsilon/prox/ortho_invariant.h"

// a .* x + b >= 0
class NonNegativeProx final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    CHECK_EQ(1, arg.f_expr().arg_size());
    affine::GetDiagonalCoefficients(arg.f_expr().arg(0), &a_, &b_, &key_);
  }

  BlockVector Apply(const BlockVector& v) override {
    BlockVector x;
    x(key_) = ((a_.cwiseProduct(v(key_)) + b_).cwiseMax(0) - b_)
              .cwiseQuotient(a_);
    return x;
  }

private:
  Eigen::VectorXd a_, b_;
  std::string key_;
};
REGISTER_PROX_OPERATOR(ProxFunction::NON_NEGATIVE, NonNegativeProx);

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
