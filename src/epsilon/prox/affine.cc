#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

// Convert 1 x n linear maps to a BlockVector
BlockVector GetLinear(const BlockMatrix& A) {
  BlockVector c;
  for (const auto& col_iter : A.data()) {
    for (const auto& row_iter: col_iter.second) {
      c(col_iter.first) = row_iter.second.impl().AsDense().transpose();
    }
  }
  return c;
}

// c'x
class AffineProx final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    const BlockMatrix& A = arg.affine_constraint().A;
    const double alpha = arg.prox_function().alpha();
    AT_ = A.Transpose();
    ATA_inv_ = (AT_*A).Inverse();
    b_ = arg.affine_constraint().b;
    if (arg.prox_function().prox_function_type() == ProxFunction::AFFINE)
      c_ = alpha*GetLinear(arg.affine_arg().A);

    VLOG(2) << "A: " << A.DebugString();
    VLOG(2) << "c: " << c_.DebugString();
  }

  BlockVector Apply(const BlockVector& v) override {
    return ATA_inv_*(AT_*(v - b_) - c_);
  }

private:
  BlockMatrix AT_, ATA_inv_;
  BlockVector b_, c_;
};

// Register twice, as same function works c = 0
REGISTER_PROX_OPERATOR(AFFINE, AffineProx);
REGISTER_PROX_OPERATOR(CONSTANT, AffineProx);
