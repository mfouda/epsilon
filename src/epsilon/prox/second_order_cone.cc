#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

// For [t X] (both of which can be scaled by a diagonal matrix), project the
// _rows_ onto the second-order cone, i.e.
//
// ||ax*x_i + bx||_2 <= at*t_i + bt_i
//
// for ax, at scalar and bx, bt vectors. By convention, "t" is the first arg.
class SecondOrderConeProx final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    InitScalar();

    // CHECK_EQ(2, arg.f_expr().arg_size());
    // double at, ax;
    // GetScalarCoefficients(arg, 0, &at, &bt_, &t_key_);
    // GetScalarCoefficients(arg, 0, &ax, &bx_, &x_key_);
    // m_ = arg.f_expr().arg(1).size().dim(0);
    // n_ = arg.f_expr().arg(1).size().dim(1);
    // a_ = at/fabs(ax);
    // bx_ /= ax;
    // bt_ /= fabs(ax);
  }

  BlockVector Apply(const BlockVector& vs) override {
    ApplyScalar(&ProjectSecondOrderCone);

    // Eigen::MatrixXd V = ToMatrix(vs(x_key_) + bx_, m_, n_);
    // Eigen::VectorXd s = vs(t_key_) + bt_/a_;
    // Eigen::VectorXd v_norm = V.rowwise().norm();
    // const double a2 = a_*a_;

    // Eigen::VectorXd alpha =
    //     ((1/(a2+1))*(a2 + a_*s.array()/v_norm.array())).cwiseMin(1).cwiseMax(0);

    // BlockVector x;
    // x(x_key_) = ToVector(alpha.asDiagonal() * V) - bx_;
    // x(t_key_) = s - bt_/a_;
    // for (int i = 0; i < m_; i++) {
    //   if (alpha(i) != 1)
    //     x(t_key_)(i) = (1/a_)*alpha(i)*v_norm(i) - bt_(i)/a_;
    // }
    // return x;
  }

private:
  double a_;
  Eigen::VectorXd bx_, bt_;
  std::string t_key_, x_key_;
  int m_, n_;
};
REGISTER_PROX_OPERATOR(ProxFunction::SECOND_ORDER_CONE, SecondOrderConeProx);
