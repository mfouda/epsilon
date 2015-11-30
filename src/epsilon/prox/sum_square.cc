
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

// ||H(x)||_2^2
class SumSquareProx final : public ProxOperator {
 public:
  void Init(const ProxOperatorArg& arg) override {
    const BlockMatrix& H = arg.affine_arg().A;
    const BlockMatrix& A = arg.affine_constraint().A;
    g_ = arg.affine_arg().b;
    HT_ = H.Transpose();
    AT_ = A.Transpose();
    F_ = (2*HT_*H + AT_*A).Inverse();
  }

  BlockVector Apply(const BlockVector& v) override {
    return F_*(AT_*v - 2*HT_*g_);
  }

 private:
  BlockVector g_;
  BlockMatrix AT_, HT_, F_;
};
REGISTER_PROX_OPERATOR(SUM_SQUARE, SumSquareProx);
