#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

// lam*||X||_{1,2}
class NormL1L2Prox final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // NORM_PQ (p: 1, q: 2)
    //   VARIABLE (X)
    const Expression& var_expr = arg.f_expr().arg(0);
    CHECK_EQ(var_expr.expression_type(), Expression::VARIABLE);
    m_ = GetDimension(var_expr, 0);
    n_ = GetDimension(var_expr, 1);
    lambda_ = arg.lambda();
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    Eigen::MatrixXd V = ToMatrix(v, m_, n_);
    Eigen::VectorXd norms = V.array().square().rowwise().sum().sqrt();
    Eigen::VectorXd s = 1 - (norms.array()/lambda_).max(1).inverse();
    return ToVector(s.asDiagonal()*V);
  }

private:
  double lambda_;
  int m_, n_;
};
REGISTER_PROX_OPERATOR(NormL1L2Prox);
