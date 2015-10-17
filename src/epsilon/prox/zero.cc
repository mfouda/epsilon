
#include "epsilon/prox/prox.h"
#include "epsilon/vector/block_matrix.h"

// f(x) = 0
// Expression tree:
// ZERO
//   VARIABLE
class ZeroProx final : public BlockProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    ATA_inv_ = arg.ATA().Inverse();
  }

  BlockVector Apply(const BlockVector& v) override {
    // NOTE(mwytock): v is already transformed by A.Transpose() before
    // being passed here.
    return ATA_inv_*v;
  }

private:
  BlockMatrix ATA_inv_;
};
REGISTER_BLOCK_PROX_OPERATOR(ZeroProx);
