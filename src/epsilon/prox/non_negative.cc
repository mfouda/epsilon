#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"

// I(alpha*x + b >= 0)
//
// Expression tree:
// INDICATOR (cone: NON_NEGATIVE)
//   SCALAR_MULTIPLE (alpha*x + b >= 0)
class NonNegativeProx final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    GetScalarAffineOperator(arg.f_expr().arg(0), arg.var_map(), &alpha_, &b_);
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    return (1/alpha_)*((alpha_*v + b_).eval().cwiseMax(0) - b_);
  }

private:
  double alpha_;
  Eigen::VectorXd b_;
};
REGISTER_PROX_OPERATOR(NonNegativeProx);
