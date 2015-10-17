
#include "epsilon/prox/prox.h"
#include "epsilon/vector/block_matrix.h"

// f(x) = 0
// Expression tree:
// ZERO
//   VARIABLE
//
// Only useful with a non trivial A matrix, in which case we solve ||Ax - v||^2
class ZeroProx final : public BlockProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    const BlockMatrix& A = arg.A();
    BlockMatrix AT = A.Transpose();
    F_ = (AT*A).Inverse()*AT;
  }

  BlockVector Apply(const BlockVector& v) override {
    return F_*v;
  }

private:
  BlockMatrix F_;
};
REGISTER_BLOCK_PROX_OPERATOR(ZeroProx);
