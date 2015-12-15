
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/block_cholesky.h"

// I(H(x) = 0)
class ZeroProx final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    const BlockMatrix& H = arg.affine_arg().A;
    const BlockVector& g = arg.affine_arg().b;
    const BlockMatrix& A = arg.affine_constraint().A;

    // [ 0   H'  A'][ x ] = [ 0 ]
    // [ H   0   0 ][ y ]   [-g ]
    // [ A   0  -I ][ z ]   [ v ]
    BlockMatrix M = H + H.Transpose() + A + A.Transpose() - A.LeftIdentity();
    VLOG(2) << "M: " << M.DebugString();
    chol_.Compute(M);
    b_ = -1*g;
  }

  BlockVector Apply(const BlockVector& v) override {
    return chol_.Solve(b_ + v);
  }

private:
  BlockCholesky chol_;
  BlockVector b_;
};
REGISTER_PROX_OPERATOR(ZERO, ZeroProx);
