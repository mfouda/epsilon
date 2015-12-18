// Orthogonally Invariant Matrix Function satisfies
//      F(X) = F(UXV') = f(eigen_values(X))

#include "epsilon/prox/ortho_invariant.h"
#include "epsilon/vector/vector_util.h"

void OrthoInvariantProx::Init(const ProxOperatorArg& arg) {
  VectorProx::Init(arg);
  m_ = arg.prox_function().arg_size(0).dim(0);
  n_ = arg.prox_function().arg_size(0).dim(1);
}

void OrthoInvariantProx::ApplyVector(
    const VectorProxInput& input,
    VectorProxOutput* output) {
  if (!init_eigen_prox_) {
    // TODO(mwytock): Fix this hack by explicit interface for lambda
    InitEigenProx(input.lambda());
    init_eigen_prox_ = true;
  }

  Eigen::MatrixXd Y = ToMatrix(input.value_vec(0), m_, n_);

  Eigen::MatrixXd U, V, R;
  Eigen::VectorXd d;
  if (add_residual_)
    R = (Y - Y.transpose()) / 2;

  if (symmetric_part_) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver((Y + Y.transpose()) / 2);
    CHECK_EQ(solver.info(), Eigen::Success);
    d = solver.eigenvalues();
    V = solver.eigenvectors();
    U = V;
  } else {
    Eigen::MatrixXd EPS = Eigen::VectorXd::Constant(n_, 1e-15).asDiagonal();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Y.transpose()*Y + EPS);
    CHECK_EQ(solver.info(), Eigen::Success)
        << MatrixDebugString(Y.transpose()*Y + EPS);
    d = solver.eigenvalues();
    V = solver.eigenvectors();
    d = d.cwiseMax(0).cwiseSqrt();
    Eigen::VectorXd dinv(d.rows());
    for (int i = 0; i < d.rows(); i++) {
      if (d(i) != 0)
        dinv(i) = 1/d(i);
      else
        dinv(i) = 0;
    }
    U = Y * V * dinv.asDiagonal();
  }
  VLOG(2) << "\nD = " << VectorDebugString(d) << "\n";

  if (epigraph_) {
    const double s = input.value(1);

    Eigen::VectorXd x_tilde;
    double t;
    ApplyEigenEpigraph(d, s, &x_tilde, &t);
    Eigen::MatrixXd X = U*x_tilde.asDiagonal()*V.transpose();

    output->set_value(0, ToVector(X));
    output->set_value(1, t);
  } else {
    Eigen::VectorXd x_tilde = ApplyEigenProx(d);
    Eigen::MatrixXd X = U*x_tilde.asDiagonal()*V.transpose();
    if (add_residual_)
      X += R;
    else if (symmetric_part_)
      X = (X + X.transpose())/2;

    output->set_value(0, ToVector(X));
  }
}

void OrthoInvariantProx::InitEigenProx(double lambda) {
  const int n = std::min(m_, n_);
  const int num_args = epigraph_ ? 2 : 1;

  eigen_prox_ = CreateProxOperator(eigen_prox_type_, epigraph_);
  alpha_ = epigraph_ ? 1 : 1/sqrt(lambda);
  ProxFunction prox_function;
  prox_function.set_prox_function_type(eigen_prox_type_);
  prox_function.set_alpha(1);
  Size* size = prox_function.add_arg_size();
  size->add_dim(n);
  size->add_dim(1);

  AffineOperator affine_arg, affine_constraint;
  for (int i = 0; i < num_args; i++) {
    std::string key = affine::arg_key(i);
    affine_arg.A(key, key) = linear_map::Identity(n);
    affine_constraint.A(key, key) = linear_map::Scalar(alpha_, n);
  }
  eigen_prox_->Init(
      ProxOperatorArg(prox_function, affine_arg, affine_constraint));
}

Eigen::VectorXd OrthoInvariantProx::ApplyEigenProx(const Eigen::VectorXd& v) {
  BlockVector input;
  input(affine::arg_key(0)) = alpha_*v;
  BlockVector output = eigen_prox_->Apply(input);
  return output(affine::arg_key(0));
}

void OrthoInvariantProx::ApplyEigenEpigraph(
    const Eigen::VectorXd& v, double s,
    Eigen::VectorXd* x, double* t) {
  BlockVector input;
  input(affine::arg_key(0)) = v;
  input(affine::arg_key(1)) = Eigen::VectorXd::Constant(1, s);
  BlockVector output = eigen_prox_->Apply(input);
  *x = output(affine::arg_key(0));
  *t = output(affine::arg_key(1))(0);
}
