#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

// a'x + b
class AffineProx final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    // BlockMatrix A;
    // BlockVector b;
    // affine::BuildAffineOperator(arg.f_expr(), "_", &A, &b);
    // CHECK_EQ(1, A.col_keys().size());
    // c_ = arg.lambda()*ToVector(A("_", *A.col_keys().begin()).impl().AsDense());
  }

  BlockVector Apply(const BlockVector& v) override {
    return v - a_;
  }

private:
  BlockVector a_;
};
REGISTER_PROX_OPERATOR(ProxFunction::AFFINE, AffineProx);
