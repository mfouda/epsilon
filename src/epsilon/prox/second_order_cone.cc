#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

void GetKeys(const BlockMatrix& H, std::string* t_key, std::string* x_key) {
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
    const BlockMatrix& H = arg.affine_arg().A;
    GetKeys(H, &t_key_, &x_key_);
    double at = linear_map::GetScalar(H(affine::arg_key(0), t_key_));
    double ax = linear_map::GetScalar(H(affine::arg_key(1), x_key_));

    const BlockVector& g = arg.affine_arg().b;
    bt_ = g(affine::arg_key(0));
    bx_ = g(affine::arg_key(1));
    m_ = arg.prox_function().m();
    n_ = arg.prox_function().n();
    a_ = at/fabs(ax);
    bx_ /= ax;
    bt_ /= fabs(ax);

    InitConstraints(arg.affine_constraint());
  }

  BlockVector Apply(const BlockVector& v) override {
    BlockVector u = AT_*v;
    Eigen::MatrixXd V = ToMatrix(u(x_key_) + bx_, m_, n_);
    Eigen::VectorXd s = u(t_key_) + bt_/a_;
    Eigen::VectorXd v_norm = V.rowwise().norm();
    const double a2 = a_*a_;

    Eigen::VectorXd alpha =
        ((1/(a2+1))*(a2 + a_*s.array()/v_norm.array())).cwiseMin(1).cwiseMax(0);

    BlockVector x;
    x(x_key_) = ToVector(alpha.asDiagonal() * V) - bx_;
    x(t_key_) = s - bt_/a_;
    for (int i = 0; i < m_; i++) {
      if (alpha(i) != 1)
        x(t_key_)(i) = (1/a_)*alpha(i)*v_norm(i) - bt_(i)/a_;
    }
    return x;
  }

private:
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
REGISTER_PROX_OPERATOR(ProxFunction::SECOND_ORDER_CONE, SecondOrderConeProx);
