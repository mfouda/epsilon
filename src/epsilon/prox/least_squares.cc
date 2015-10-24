
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
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
    BlockVector b;
    affine::BuildAffineOperator(arg.f_expr().arg(0).arg(0), "f", &C_, &b);
    const BlockMatrix& A = arg.A();

    VLOG(2) << "A: " << A.DebugString();
    VLOG(2) << "C: " << C_.DebugString();
    VLOG(2) << "b: " << b.DebugString();

    rho_ = 1/(2*arg.lambda());
    CT_ = C_.Transpose();
    AT_ = A.Transpose();

    // Use Gaussian elimination to solve the system:
    //
    // [ pA'A  C'][ x ] = [ pA'v ]
    // [ C    -I ][ y ]   [ b    ]
    //
    // where p = 1/(2*lambda). We assume that A'A is diagonal so that the
    // computaitonal time is dominated by C.
    if (C_.m() <= C_.n()) {
      fat_ = true;
      D_inv_ = (rho_*AT_*A).Inverse();
      H_inv_ = (C_.LeftIdentity() + C_*D_inv_*CT_).Inverse();
      g_ = D_inv_*CT_*(H_inv_*b);
    } else {
      fat_ = false;
      H_inv_ = (rho_*AT_*A + CT_*C_).Inverse();
      g_ = H_inv_*(CT_*b);
    }
  }

  BlockVector Apply(const BlockVector& v) override {
    if (fat_) {
      BlockVector w = D_inv_*(AT_*v);
      return rho_*(w - D_inv_*(CT_*(H_inv_*(C_*w)))) - g_;
    } else {
      return rho_*(H_inv_*(AT_*v)) - g_;
    }
  }

 private:
  bool fat_;
  double rho_;
  BlockMatrix D_inv_, H_inv_;
  BlockMatrix C_, CT_, AT_;

  BlockMatrix F_;
  BlockVector g_;
};
REGISTER_BLOCK_PROX_OPERATOR(LeastSquaresProx);
