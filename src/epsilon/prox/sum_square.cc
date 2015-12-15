
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"
#include "epsilon/vector/block_cholesky.h"

// ||H(x)||_2^2
class SumSquareProx final : public ProxOperator {
 public:
  void Init(const ProxOperatorArg& arg) override {
    const BlockMatrix& H = arg.affine_arg().A;
    const BlockVector& g = arg.affine_arg().b;
    const BlockMatrix& A = arg.affine_constraint().A;
    const double alpha = sqrt(2*arg.prox_function().alpha());

    // [ 0   H'  A'][ x ] = [ 0 ]
    // [ H  -I   0 ][ y ]   [-g ]
    // [ A   0  -I ][ z ]   [ v ]
    // TODO(mwytock): Cholesky factorization should not require full matrix
    // since it is symmetric.
    BlockMatrix M = alpha*(H + H.Transpose()) + (A + A.Transpose())
                    - H.LeftIdentity() - A.LeftIdentity();
    VLOG(2) << "M: " << M.DebugString();
    chol_.Compute(M);
    b_ = -alpha*g;
  }

  BlockVector Apply(const BlockVector& v) override {
    return chol_.Solve(b_ + v);
  }

 private:
  BlockCholesky chol_;
  BlockVector b_;
};
REGISTER_PROX_OPERATOR(SUM_SQUARE, SumSquareProx);
