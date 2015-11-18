#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

// a'x + b
class AffineProx final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    CHECK_EQ(1, GetDimension(arg.f_expr()));
    CHECK_EQ(1, arg.f_expr().arg_size());

    BlockMatrix A;
    BlockVector b;
    affine::BuildAffineOperator(arg.f_expr().arg(0), "_", &A, &b);

    // Get the coefficients of the linear function
    for (const auto& col_iter : A.data()) {
      for (const auto& row_iter : col_iter.second) {
        a_(col_iter.first) = row_iter.second.impl().AsDense();
      }
    }
    a_ *= arg.lambda();
  }

  BlockVector Apply(const BlockVector& v) override {
    return v - a_;
  }

private:
  BlockVector a_;
};
REGISTER_PROX_OPERATOR(ProxFunction::AFFINE, AffineProx);
