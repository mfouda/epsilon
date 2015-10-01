#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"

// c'x
class LinearProx final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    DynamicMatrix A = DynamicMatrix::Zero(1, arg.var_map().n());
    DynamicMatrix b = DynamicMatrix::Zero(1, 1);
    BuildAffineOperator(arg.f_expr(), arg.var_map(), &A, &b);
    c_ = arg.lambda()*A.AsDense().transpose();
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    return v - c_;
  }

private:
  Eigen::VectorXd c_;
};
REGISTER_PROX_OPERATOR(LinearProx);
