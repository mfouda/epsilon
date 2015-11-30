#include "epsilon/prox/vector.h"
#include "epsilon/vector/vector_util.h"

void VectorProx::Init(const ProxOperatorArg& arg) {
  InitArgs(arg.affine_arg());
  InitConstraints(arg.affine_constraint());
  InitVector(n_, lambda_);

  VLOG(2) << "AT: " << AT_.DebugString();
  VLOG(2) << "lambda: " << lambda_ << ", alpha: " << alpha_;
  VLOG(2) << "b: " << VectorDebugString(b_);
}

void VectorProx::InitArgs(const AffineOperator& f) {
  // Assumes single argument and single variable
  const BlockMatrix& H = f.A;
  const BlockVector& g = f.b;
  CHECK(H.row_keys().size() == 1 && H.col_keys().size() == 1);

  key_ = *H.col_keys().begin();
  n_ = H(affine::arg_key(0), key_).impl().n();
  alpha_ = linear_map::GetScalar(H(affine::arg_key(0), key_));
  b_ = g.has_key(affine::arg_key(0)) ? g(affine::arg_key(0))
       : Eigen::VectorXd::Zero(n_);

  lambda_ = 1;
}

void VectorProx::InitConstraints(const AffineOperator& f) {
  // A'A must be scalar
  const BlockMatrix& A = f.A;
  AT_ = A.Transpose();
  double alpha = linear_map::GetScalar((AT_*A)(key_, key_));

  // Scale lambda and A' by alpha
  lambda_ /= alpha;
  AT_ = (1/alpha)*AT_;
}

BlockVector VectorProx::Apply(const BlockVector& v) {
  BlockVector x;
  // Apply the composition rules
  x(key_) = (ApplyVector(alpha_*(AT_*v)(key_) + b_) - b_)/alpha_;
  return x;
}
