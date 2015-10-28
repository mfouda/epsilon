
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

// lam*||Ax + b||_2^2 + (1/2)||x - v||
//
// Expression tree:
// SUM
//   POWER (p: 2)
//     AFFINE (x)
class LeastSquaresProx final : public BlockProxOperator {
 public:
  void Init(const ProxOperatorArg& arg) override {
    BlockMatrix A;
    BlockVector b;
    affine::BuildAffineOperator(arg.f_expr().arg(0).arg(0), "f", &A, &b);

    VLOG(2) << "A: " << A.DebugString();
    VLOG(2) << "b: " << b.DebugString();

    BlockMatrix AT = A.Transpose();
    rho_ = 1/(2*arg.lambda());

    // Use Gaussian elimination to solve the system:
    //
    // [ pI  A'][ x ] = [ pv ]
    // [ A  -I ][ y ]   [ -b ]
    if (A.m() <= A.n()) {
      fat_ = true;
      AAT_inv_ = (A*AT + rho_*A.LeftIdentity()).Inverse();
      q_ = AT*AAT_inv_*b;
      A_ = A;
      AT_ = AT;
    } else {
      fat_ = false;
      ATA_inv_ = (AT*A + rho_*A.RightIdentity()).Inverse();
      q_ = ATA_inv_*AT*b;
    }
  }

  BlockVector Apply(const BlockVector& v) override {
    if (fat_) {
      return v - AT_*(AAT_inv_*(A_*v)) - q_;
    } else {
      return ATA_inv_*(rho_*v) - q_;
    }
  }

 private:
  bool fat_;
  double rho_;
  BlockVector q_;
  BlockMatrix A_, AT_, ATA_inv_, AAT_inv_;
};
REGISTER_BLOCK_PROX_OPERATOR(LeastSquaresProx);
