#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/block_cholesky.h"
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
    const BlockVector& b = arg.affine_constraint().b;
    const double alpha = arg.prox_function().alpha();

    BlockVector c;
    if (arg.prox_function().prox_function_type() == ProxFunction::AFFINE)
      c = alpha*GetLinear(arg.affine_arg().A);

    VLOG(2) << "A: " << A.DebugString();
    VLOG(2) << "b: " << b.DebugString();
    VLOG(2) << "c: " << c.DebugString();

    BlockMatrix M = A + A.Transpose() - A.LeftIdentity();
    chol_.Compute(M);
    g_ = -1*b - c;
  }

  BlockVector Apply(const BlockVector& v) override {
    return chol_.Solve(g_ + v);
  }

private:
  BlockCholesky chol_;
  BlockVector g_;
};

// Register twice, as same function works c = 0
REGISTER_PROX_OPERATOR(AFFINE, AffineProx);
REGISTER_PROX_OPERATOR(CONSTANT, AffineProx);
