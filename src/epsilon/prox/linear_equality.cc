
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
    BlockMatrix C;
    BlockVector b;
    affine::BuildAffineOperator(arg.f_expr().arg(0), "f", &C, &b);

    VLOG(2) << "A: " << A.DebugString();
    VLOG(2) << "C: " << C.DebugString();
    VLOG(2) << "b: " << b.DebugString();

    BlockMatrix AT = A.Transpose();
    BlockMatrix CT = C.Transpose();

    // Use Gaussian elimination to solve the system:
    //
    // [ A'A  C' ][ x ] = [  A'v ]
    // [ C    0  ][ y ]   [ -b   ]
    BlockMatrix D_inv = (AT*A).Inverse();
    BlockMatrix H_inv = (C*D_inv*CT).Inverse();
    F_ = D_inv*(C.RightIdentity() - CT*H_inv*C*D_inv)*AT;
    g_ = D_inv*(CT*(H_inv*b));
  }

  BlockVector Apply(const BlockVector& v) override {
    return F_*v - g_;
  }

private:
  BlockMatrix F_;
  BlockVector g_;
};
REGISTER_BLOCK_PROX_OPERATOR(LinearEqualityProx);
