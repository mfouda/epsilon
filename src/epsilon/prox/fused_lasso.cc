
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

extern "C" {
void tf_dp (int n, double *y, double lam, double *beta);
}

// lam*||x[2:] - x[:-1]||_1
//
// Expression tree:
// NORM_P (p: 1)
//   ADD
//     INDEX (key: [1:N, 0:1])
//       VARIABLE (variable_id: x)
//     NEGATE
//       INDEX (key: [0:N-1, 0:1])
//         VARIABLE (variable_id: cvxpy:0)
class FusedLassoProx final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    n_ = GetDimension(arg.f_expr().arg(0).arg(0).arg(0));
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    Eigen::VectorXd x(n_);
    tf_dp(n_, const_cast<double*>(v.data()), lambda_, x.data());
    return x;
  }

private:
  // Params
  double lambda_;
  int n_;
};
