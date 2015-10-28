
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/block_matrix.h"

// I(Ax + b = 0)
// Expression tree:
// INDICATOR (cone: ZERO)
//   AFFINE
class LinearEqualityProx final : public BlockProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    BlockMatrix A;
    BlockVector b;
    affine::BuildAffineOperator(arg.f_expr().arg(0), "f", &A, &b);

    graph_form_ = InitGraphForm();
    if (graph_form_)
      return;

    // Standard case, solve:
    // x  + A'y = v
    // Ax       = -b
    A_ = A;
    AT_ = A.Transpose();
    AAT_inv_ = (A*AT_).Inverse();
    ATb_ = AT_*b;

  }

  BlockVector Apply(const BlockVector& v) override {
    // if (graph_form_) {
    //   //
    // }

    return v - AT_*(AAT_inv_*(A_*v)) - ATb_;
  }

private:
  bool InitGraphForm() {
    return false;
  }

  bool graph_form_;
  BlockMatrix AT_, AAT_inv_, A_;
  BlockVector ATb_;
};
REGISTER_BLOCK_PROX_OPERATOR(LinearEqualityProx);
