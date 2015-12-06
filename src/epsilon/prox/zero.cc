
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/block_matrix.h"

// I(H(x) = 0)
class ZeroProx final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    H_ = arg.affine_arg().A;
    g_ = arg.affine_arg().b;
    A_ = arg.affine_constraint().A;
    b_ = arg.affine_constraint().b;

    // Use Gaussian elimination to solve the system
    // [ A'A  H' ][ x ] = [ A'(v - b) ]
    // [ H    0  ][ y ]   [ -g        ]
    AT_ = A_.Transpose();
    HT_ = H_.Transpose();
    ATA_inv_ = (AT_*A_).Inverse();
    HHT_inv_ = (H_*ATA_inv_*HT_).Inverse();

    VLOG(2) << "H:" << H_.DebugString();
    VLOG(2) << "g:" << g_.DebugString();
    VLOG(2) << "A:" << A_.DebugString();
  }

  BlockVector Apply(const BlockVector& v) override {
    BlockVector u = ATA_inv_*(AT_*(v - b_));
    BlockVector y = HHT_inv_*(H_*u + g_);
    return u - ATA_inv_*(HT_*y);
  }

private:
  BlockMatrix A_, AT_, H_, HT_, ATA_inv_, HHT_inv_;
  BlockVector b_, g_;
};
REGISTER_PROX_OPERATOR(ZERO, ZeroProx);
