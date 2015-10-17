
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
    BlockMatrix CT = C.Transpose();
    BlockMatrix AT = A.Transpose();

    // Use Gaussian elimination to solve the system:
    //
    // [ pA'A  C'][ x ] = [ pA'v ]
    // [ C    -I ][ y ]   [ b    ]
    //
    // where p = 1/(2*lambda). We assume that A'A is diagonal so that the
    // computaitonal time is dominated by C.
    if (C.m() <= C.n()) {
      BlockMatrix D_inv = (rho*AT*A).Inverse();
      BlockMatrix H_inv = (C.LeftIdentity() + C*D_inv*CT).Inverse();
      F_ = rho*D_inv*(C.RightIdentity() - CT*H_inv*C*D_inv)*AT;
      g_ = D_inv*CT*(H_inv*b);
    } else {
      BlockMatrix H_inv =(rho*AT*A + CT*C).Inverse();
      F_ = rho*H_inv*AT;
      g_ = H_inv*(CT*b);
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
