#include "epsilon/prox/vector_prox.h"
#include "epsilon/vector/vector_util.h"

double GetScalar(const BlockMatrix& A) {
  bool first = true;
  double alpha;
  for (const auto& col_iter : A.data()) {
    if (col_iter.second.size() != 1 ||
        col_iter.first != col_iter.second.begin()->first)
      LOG(FATAL) << "Not block diagonal";

    if (first) {
      alpha = linear_map::GetScalar(col_iter.second.begin()->second);
      first = false;
    } else {
      CHECK_EQ(alpha, linear_map::GetScalar(col_iter.second.begin()->second));
    }
  }

  return alpha;
}

void VectorProx::Init(const ProxOperatorArg& arg) {
  // transform
  //   argmin_x f(Hx + g) + 1/2||Ax - v||^2 to
  // to
  //   argmin_x lambda*f(x) + 1/2||x - (M*v + g)||^2

  const BlockMatrix& A = arg.affine_constraint().A;
  const BlockMatrix& H = arg.affine_arg().A;
  g_ = arg.affine_arg().b;
  H_inv_ = H.Inverse();

  BlockMatrix M = A*H_inv_;
  MT_ = M.Transpose();

  // Must be a scalar/diagonal matrix
  // TODO(mwtyock): Support diagonal
  input_.lambda_ = 1/GetScalar(MT_*M);
  input_.lambda_vec_ = Eigen::VectorXd::Constant(A.n(), input_.lambda_);
  input_.elementwise_ = false;
  MT_ = input_.lambda_*MT_;

  VLOG(2) << "MT: " << MT_.DebugString();
  VLOG(2) << "H_inv: " << H_inv_.DebugString();
  VLOG(2) << "g: " << g_.DebugString();
}

void VectorProx::PreProcessInput(const BlockVector& v) {
  input_.v_ = MT_*v + g_;
}

BlockVector VectorProx::PostProcessOutput() {
  return H_inv_*(output_.x_ - g_);
}

BlockVector VectorProx::Apply(const BlockVector& v) {
  PreProcessInput(v);
  ApplyVector(input_, &output_);
  return PostProcessOutput();
}

double VectorProxInput::lambda() const {
  CHECK(!elementwise_);
  return lambda_;
}

const Eigen::VectorXd& VectorProxInput::lambda_vec() const {
  return lambda_vec_;
}

double VectorProxInput::value(int i) const {
  const Eigen::VectorXd& val = value_vec(i);
  CHECK_EQ(1, val.size());
  return val(0);
}

const Eigen::VectorXd& VectorProxInput::value_vec(int i) const {
  return v_(affine::arg_key(i));
}

void VectorProxOutput::set_value(int i, double x) {
  x_(affine::arg_key(i)) = Eigen::VectorXd::Constant(1, x);
}

void VectorProxOutput::set_value(int i, const Eigen::VectorXd& x) {
  x_(affine::arg_key(i)) = x;
}
