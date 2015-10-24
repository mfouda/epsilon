
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/block_matrix.h"

// I(Cx + b == 0) + (1/2)||Ax - v||^2
// Expression tree:
// INDICATOR (cone: ZERO)
//   AFFINE
class LinearEqualityProx final : public BlockProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    const BlockMatrix& A = arg.A();
    BlockVector b;
    affine::BuildAffineOperator(arg.f_expr().arg(0), "f", &C_, &b);

    VLOG(2) << "A: " << A.DebugString();
    VLOG(2) << "C: " << C_.DebugString();
    VLOG(2) << "b: " << b.DebugString();

    AT_ = A.Transpose();
    CT_ = C_.Transpose();

    // Use Gaussian elimination to solve the system:
    //
    // [ A'A  C' ][ x ] = [  A'v ]
    // [ C    0  ][ y ]   [ -b   ]
    D_inv_ = (AT_*A).Inverse();
    H_inv_ = (C_*D_inv_*CT_).Inverse();
    g_ = D_inv_*(CT_*(H_inv_*b));
  }

  BlockVector Apply(const BlockVector& v) override {
    BlockVector w = D_inv_*(AT_*v);
    return w - CT_*(H_inv_*(C_*w)) - g_;
  }

private:
  BlockMatrix D_inv_, H_inv_, AT_, C_, CT_;
  BlockVector g_;
};
REGISTER_BLOCK_PROX_OPERATOR(LinearEqualityProx);
