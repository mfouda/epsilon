#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

// lam*||x||_2
class NormL2Prox final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // NORM_P (p: 2)
    //   VARIABLE (x)
    lambda_ = arg.lambda();
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    const double v_norm = v.norm();
    if (v_norm >= lambda_) {
      return (1 - lambda_/v_norm)*v;
    } else {
      return Eigen::VectorXd::Zero(v.rows());
    }
  }

private:
  double lambda_;
};
REGISTER_PROX_OPERATOR(NormL2Prox);

// I(||x||_2 <= t)
class NormL2Epigraph final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // INDICATOR (cone: NON_NEGATIVE)
    //   VARIABLE (t)
    //   NORM_P (p: 2)
    //     VARIABLE (x)

    A_ = GetSparseAffineOperator(arg.f_expr().arg(0), arg.var_map());
    B_ = GetSparseAffineOperator(arg.f_expr().arg(1).arg(0), arg.var_map());
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& vs) override {
    Eigen::VectorXd v = B_*vs;
    const double s = (A_*vs)(0);
    const double v_norm = v.norm();

    if (v_norm <= -s) {
      return Eigen::VectorXd::Zero(vs.rows());
    } else if (v_norm <= s) {
      return vs;
    } else {
      return 0.5*(1 + s/v_norm)*(
          B_.transpose()*v +
          static_cast<Eigen::MatrixXd>(A_.transpose())*v_norm);
    }
  }

private:
  SparseXd A_, B_;
};
REGISTER_PROX_OPERATOR(NormL2Epigraph);
