
#include "epsilon/prox/elementwise.h"
#include "epsilon/vector/vector_util.h"

void ElementwiseProx::Init(const ProxOperatorArg& arg) {
  InitArgs(arg.affine_arg());
  InitConstraints(arg.affine_constraint());

  VLOG(2) << "AT: " << AT_.DebugString();
  VLOG(2) << "lambda: " << VectorDebugString(lambda_);
  VLOG(2) << "a: " << VectorDebugString(a_);
  VLOG(2) << "b: " << VectorDebugString(b_);
}

void ElementwiseProx::InitArgs(const AffineOperator& f) {
  // Assumes single argument and single variable
  const BlockMatrix& H = f.A;
  const BlockVector& g = f.b;
  CHECK(H.row_keys().size() == 1 && H.col_keys().size() == 1);

  key_ = *H.col_keys().begin();
  a_ = linear_map::GetDiagonal(H(affine::arg_key(0), key_));
  b_ = g.has_key(affine::arg_key(0)) ? g(affine::arg_key(0))
       : Eigen::VectorXd::Zero(a_.rows());
  lambda_ = Eigen::VectorXd::Constant(a_.rows(), 1);
}

void ElementwiseProx::InitConstraints(const AffineOperator& f) {
  // A'A must be diagonal
  const BlockMatrix& A = f.A;
  AT_ = A.Transpose();
  const Eigen::VectorXd alpha = linear_map::GetDiagonal((AT_*A)(key_, key_));

  // Scale lambda and A' by alpha
  lambda_.array() /= alpha.array();
  BlockMatrix D;
  D(key_, key_) = linear_map::Diagonal(alpha.cwiseInverse());
  AT_ = D*AT_;
}

BlockVector ElementwiseProx::Apply(const BlockVector& v) {
  BlockVector x;
  // Apply the composition rules
  x(key_) = (ApplyElementwise(
      lambda_,
      a_.cwiseProduct((AT_*v)(key_)) + b_) - b_).cwiseQuotient(a_);
  return x;
}
