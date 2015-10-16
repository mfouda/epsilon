
#include "epsilon/affine/affine.h"
#include "epsilon/affine/affine_matrix.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/block_matrix.h"

// I(Ax + b == 0)
// Expression tree:
// INDICATOR (cone: ZERO)
//   AFFINE
class LinearEqualityProx final : public BlockProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    CHECK_EQ(1, arg.f_expr().arg_size());
    affine::BuildAffineOperator(arg.f_expr().arg(0), "f", &A_, &b_);
    AT_ = A_.Transpose();
    AAT_inv_ = (A_*AT_).Inverse();

    VLOG(2) << "A:\n" << A_.DebugString();
    VLOG(2) << "b:\n" << b_.DebugString();
  }

  BlockVector Apply(const BlockVector& v) override {
    return v - AT_*(AAT_inv_*(A_*v + b_));
  }

private:
  BlockMatrix A_, AT_, AAT_inv_;
  BlockVector b_;
};
REGISTER_BLOCK_PROX_OPERATOR(LinearEqualityProx);
