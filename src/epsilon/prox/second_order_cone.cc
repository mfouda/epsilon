#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

void GetArgKeys(const BlockMatrix& H, std::string* t_key, std::string* x_key) {
  for (const auto& col_iter : H.data()) {
    for (const auto& row_iter : col_iter.second) {
      if (row_iter.first == affine::arg_key(0)) {
        *t_key = col_iter.first;
      } else if (row_iter.first == affine::arg_key(1)) {
        *x_key = col_iter.first;
      } else {
        LOG(FATAL) << "Unknown row key " << row_iter.first;
      }
    }
  }
}

// For [t X] (both of which can be scaled by a diagonal matrix), project the
// _rows_ onto the second-order cone, i.e.
//
// ||ax*x_i + bx||_2 <= at*t_i + bt_i
//
// for ax, at scalar and bx, bt vectors. By convention, "t" is the first arg.
//
// TODO(mwytock): Refactor this once we have better base classes for
// multi-argument functions.
class SecondOrderConeProx final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    m_ = arg.prox_function().m();
    n_ = arg.prox_function().n();
    InitArgs(arg.affine_arg());
    InitConstraints(arg.affine_constraint());

    VLOG(2) << "AT: " << AT_.DebugString();
    VLOG(2) << "t_key: " << t_key_ << ", x_key: " << x_key_;
    VLOG(2) << "cones m=" << m_ << ", dimension n=" << n_;
    VLOG(2) << "a: " << a_;
    VLOG(2) << "bt: " << VectorDebugString(bt_);
    VLOG(2) << "bx: " << VectorDebugString(bx_);
  }

  BlockVector Apply(const BlockVector& v) override {
    BlockVector u = AT_*v;
    Eigen::MatrixXd X = ToMatrix(u(x_key_) + bx_, m_, n_);
    Eigen::VectorXd t = u(t_key_) + bt_/a_;
    ApplyProjection(&X, &t, a_);
    BlockVector x;
    x(x_key_) = ToVector(X) - bx_;
    x(t_key_) = t - bt_/a_;
    return x;
  }

private:
  void ApplyProjection(Eigen::MatrixXd* X, Eigen::VectorXd* t, double beta) {
    Eigen::VectorXd v_norm = X->rowwise().norm();
    const double beta2 = beta*beta;
    Eigen::VectorXd alpha =
        ((1/(beta2+1))*(beta2 + beta*t->array()/v_norm.array()));

    for (int i = 0; i < m_; i++) {
      if (isnan(alpha(i)) || alpha(i) > 1) {
        alpha(i) = 1;
      } else if (alpha(i) < 0) {
        alpha(i) = 0;
        (*t)(i) = 0;
      } else {
        (*t)(i) = (1/beta)*alpha(i)*v_norm(i);
      }
    }
    *X = alpha.asDiagonal() * (*X);
  }

  void InitArgs(const AffineOperator& f) {
    const BlockMatrix& H = f.A;
    GetArgKeys(H, &t_key_, &x_key_);
    double at = linear_map::GetScalar(H(affine::arg_key(0), t_key_));
    double ax = linear_map::GetScalar(H(affine::arg_key(1), x_key_));

    const BlockVector& g = f.b;
    bt_ = g.Get(affine::arg_key(0), m_);
    bx_ = g.Get(affine::arg_key(1), m_*n_);
    a_ = at/fabs(ax);
    bx_ /= ax;
    bt_ /= fabs(ax);
  }

  void InitConstraints(const AffineOperator& f) {
    // A'A must be scalar
    const BlockMatrix& A = f.A;
    AT_ = A.Transpose();
    BlockMatrix ATA = AT_*A;
    const double alphat = linear_map::GetScalar(ATA(t_key_, t_key_));
    const double alphax = linear_map::GetScalar(ATA(x_key_, x_key_));
    CHECK_EQ(alphat, alphax) << "A'A not scalar matrix";

    BlockMatrix D;
    D(x_key_, x_key_) = linear_map::Scalar(1/alphat, m_*n_);
    D(t_key_, t_key_) = linear_map::Scalar(1/alphat, m_);
    AT_ = D*AT_;
  }

  BlockMatrix AT_;
  double a_;
  Eigen::VectorXd bx_, bt_;
  std::string t_key_, x_key_;
  int m_, n_;
};
REGISTER_PROX_OPERATOR(SECOND_ORDER_CONE, SecondOrderConeProx);
