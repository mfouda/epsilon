#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

Eigen::VectorXd GetElementwiseMultiply(
    const Expression& expr,
    const VariableOffsetMap& var_map) {
  SparseXd A = GetSparseAffineOperator(expr, var_map);
  CHECK(IsDiagonal(A));
  return A.diagonal();
}

// lam*||a .* x||_1
class NormL1Prox final : public ProxOperator {
 public:
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // NORM_P (p: 1)
    //   VARIABLE (x)
    t_ = arg.lambda()*GetElementwiseMultiply(
        arg.f_expr().arg(0), arg.var_map());
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    return (( v.array()-t_.array()).max(0) -
            (-v.array()-t_.array()).max(0));
  }

 private:
  // Elementwise soft-thresholding values
  Eigen::VectorXd t_;
};
REGISTER_PROX_OPERATOR(NormL1Prox);
