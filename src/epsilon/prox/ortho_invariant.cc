// Orthogonally Invariant Matrix Function satisfies
//      F(X) = F(UXV') = f(eigen_values(X))

#include "epsilon/prox/ortho_invariant.h"
#include "epsilon/vector/vector_util.h"

void OrthoInvariantProx::Init(const ProxOperatorArg& arg) {
  CHECK_EQ(1, arg.prox_function().arg_size_size());
  CHECK_EQ(2, arg.prox_function().arg_size(0).dim_size());
  m_ = arg.prox_function().arg_size(0).dim(0);
  n_ = arg.prox_function().arg_size(0).dim(1);
  lambda_ = 1;
  InitArgs(arg.affine_arg());
  InitConstraints(arg.affine_constraint());
  eigen_prox_->InitElementwise(
      Eigen::VectorXd::Constant(std::min(m_, n_), lambda_));

  VLOG(2) << "AT: "     << AT_.DebugString();
  VLOG(2) << "lambda: " << lambda_;
  VLOG(2) << "B:\n "    << MatrixDebugString(B_);
}

void OrthoInvariantProx::InitArgs(const AffineOperator& f) {
  const BlockMatrix& H = f.A;
  const BlockVector& g = f.b;
  CHECK(H.row_keys().size() == 1 && H.col_keys().size() == 1);

  // Assumes no scaling/translation
  key_ = *H.col_keys().begin();

  // TODO(mwytock): Handle scaling and translation of args
  const double alpha = linear_map::GetScalar(H(affine::arg_key(0), key_));
  CHECK_EQ(1, alpha);

  B_ = g.has_key(affine::arg_key(0)) ?
       ToMatrix(g(affine::arg_key(0)), m_, n_) :
       Eigen::MatrixXd::Zero(m_, n_);
  CHECK(B_.isZero());
}

void OrthoInvariantProx::InitConstraints(const AffineOperator& f) {
  // A'A must be scalar
  const BlockMatrix& A = f.A;
  AT_ = A.Transpose();

  const double alpha = linear_map::GetScalar((AT_*A)(key_, key_));
  lambda_ /= alpha;
  AT_ = (1/alpha)*AT_;
}

BlockVector OrthoInvariantProx::Apply(const BlockVector& v) {
  BlockVector x;
  x(key_) = ToVector(ApplyOrthoInvariant(ToMatrix((AT_*v)(key_), m_, n_)));
  return x;
}

Eigen::MatrixXd OrthoInvariantProx::ApplyOrthoInvariant(const Eigen::MatrixXd& Y) {
  Eigen::VectorXd d;
  Eigen::MatrixXd U, V, R;

  if (add_non_symmetric_)
    R = (Y - Y.transpose()) / 2;

  if (symmetric_) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver((Y + Y.transpose()) / 2);
    CHECK_EQ(solver.info(), Eigen::Success);
    d = solver.eigenvalues();
    V = solver.eigenvectors();
    U = V;
  } else {
    Eigen::MatrixXd EPS = Eigen::VectorXd::Constant(n_, 1e-15).asDiagonal();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Y.transpose()*Y + EPS);
    CHECK_EQ(solver.info(), Eigen::Success);
    d = solver.eigenvalues();
    V = solver.eigenvectors();
    d = d.cwiseSqrt();
    U = V * V * d.asDiagonal().inverse();
  }

  VLOG(2) << "\nD = " << VectorDebugString(d) << "\n";
  Eigen::VectorXd x_tilde = eigen_prox_->ApplyElementwise(d);
  Eigen::MatrixXd X = U*x_tilde.asDiagonal()*V.transpose();
  if (add_non_symmetric_)
    X += R;

  return X;
}

// Eigen::VectorXd OrthoInvariantEpigraph::Apply(const Eigen::VectorXd& sy) {
//     int n = (int)std::sqrt(sy.rows()-1);

//     double s = sy(0);
//     Eigen::MatrixXd Y = ToMatrix(sy.tail(n*n), n, n);
//     Eigen::VectorXd d;
//     Eigen::MatrixXd U, V;
//     if(symm_part_) {
//       Y = (Y + Y.transpose()) / 2;
//       Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Y);
//       CHECK_EQ(solver.info(), Eigen::Success);
//       d = solver.eigenvalues();
//       U = solver.eigenvectors();
//       V = U;
//     } else {
//       Eigen::MatrixXd EPS = Eigen::VectorXd::Constant(n, 1e-15).asDiagonal();
//       Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Y.transpose()*Y + EPS);
//       CHECK_EQ(solver.info(), Eigen::Success);
//       d = solver.eigenvalues();
//       V = solver.eigenvectors();
//       d = d.cwiseSqrt();
//       U = Y * V * d.asDiagonal().inverse();
//     }

//     VLOG(2) << "\nD = " << VectorDebugString(d) << "\n";

//     Eigen::VectorXd f_sv(1+n);
//     f_sv(0) = s;
//     f_sv.tail(n) = d;
//     Eigen::VectorXd f_tx = f_->Apply(f_sv);

//     double t = f_tx(0);
//     Eigen::VectorXd x_tilde = f_tx.tail(n);
//     Eigen::MatrixXd X = U*x_tilde.asDiagonal()*V.transpose();
//     VectorXd tx(1+n*n);
//     tx(0) = t;
//     tx.tail(n*n) = ToVector(X);

//     return tx;
// }
