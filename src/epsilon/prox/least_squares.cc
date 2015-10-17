
#include "epsilon/affine/affine.h"
#include "epsilon/affine/affine_matrix.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

// lam*||Cx + b||_2^2 + (1/2)||Ax - v||
//
// Expression tree:
// SUM
//   POWER (p: 2)
//     AFFINE (x)
class LeastSquaresProx final : public BlockProxOperator {
 public:
  void Init(const ProxOperatorArg& arg) override {
    BlockMatrix C;
    BlockVector b;
    affine::BuildAffineOperator(arg.f_expr().arg(0).arg(0), "f", &C, &b);

    const BlockMatrix& A = arg.A();
    const double rho = 1/(2*arg.lambda());

    // Use Gaussian elimination / matrix inversion lemma depending on the size
    // of C. We assume that ATA is diagonal.
    if (C.m() <= C.n()) {
      // y = (I + C*(rho*A'A)^{-1}*C')^{-1}b
      // x = rho*A'v - C'y
      BlockMatrix ATA_inv = rho*(A.Transpose()*A).Inverse();
      BlockVector y = (C.LeftIdentity() + C*ATA_inv*C.Transpose()).Inverse()*b;
      F_ = ATA_inv*rho*A.Transpose();
      g_ = ATA_inv*C.Transpose()*y;
    } else {
      // x = (rho*A'A + C'C)^{-1} * (rho*A'v - b)
      BlockMatrix ATA_CTC_inv =(rho*A.Transpose()*A + C.Transpose()*C).Inverse();
      F_ = ATA_CTC_inv*rho*A.Transpose();
      g_ = ATA_CTC_inv*b;
    }
  }

  BlockVector Apply(const BlockVector& v) override {
    return F_*v - g_;
  }

 private:
  BlockMatrix F_;
  BlockVector g_;
};
REGISTER_BLOCK_PROX_OPERATOR(LeastSquaresProx);
