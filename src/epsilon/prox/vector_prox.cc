#include "epsilon/prox/vector_prox.h"
#include "epsilon/vector/vector_util.h"

bool GetScalar(const BlockMatrix& A, double* alpha) {
  bool first = true;
  for (const auto& col_iter : A.data()) {
    if (col_iter.second.size() != 1 ||
        col_iter.first != col_iter.second.begin()->first)
      return false;

    const linear_map::LinearMap& Ai = col_iter.second.begin()->second;
    if (Ai.impl().type() != linear_map::SCALAR_MATRIX)
      return false;

    const double alpha_i = linear_map::GetScalar(Ai);
    if (first) {
      *alpha = alpha_i;
      first = false;
    } else if (*alpha != alpha_i) {
      return false;
    }
  }

  return true;
}

bool GetDiagonal(const BlockMatrix& A, Eigen::VectorXd* alpha) {
  bool first = true;
  for (const auto& col_iter : A.data()) {
    if (col_iter.second.size() != 1 ||
        col_iter.first != col_iter.second.begin()->first)
      return false;

    const linear_map::LinearMap& Ai = col_iter.second.begin()->second;
    if (Ai.impl().type() != linear_map::SCALAR_MATRIX &&
        Ai.impl().type() != linear_map::DIAGONAL_MATRIX)
      return false;

    const Eigen::VectorXd alpha_i = linear_map::GetDiagonal(Ai);
    if (first) {
      *alpha = alpha_i;
      first = false;
    } else if (*alpha != alpha_i) {
      return false;
    }
  }

  return true;
}

bool VectorProx::InitScalar(const ProxOperatorArg& arg) {
  const double alpha = arg.prox_function().alpha();
  const BlockMatrix& H = arg.affine_arg().A;
  const BlockMatrix& A = arg.affine_constraint().A;
  BlockMatrix HT = H.Transpose();
  BlockMatrix AT = A.Transpose();

  double beta, gamma;
  if (!GetScalar(HT*H, &beta) || !GetScalar(H*AT*A*HT, &gamma))
    return false;

  B_ = (beta/gamma)*H*AT;
  C_ = (1/beta)*HT;
  input_.lambda_ = alpha*beta*beta/gamma;
  input_.lambda_vec_ = Eigen::VectorXd::Constant(A.n(), input_.lambda_);
  input_.elementwise_ = false;
  return true;
}

bool VectorProx::InitDiagonal(const ProxOperatorArg& arg) {
  const double alpha = arg.prox_function().alpha();
  const BlockMatrix& H = arg.affine_arg().A;
  const BlockMatrix& A = arg.affine_constraint().A;
  BlockMatrix HT = H.Transpose();
  BlockMatrix AT = A.Transpose();

  Eigen::VectorXd beta, gamma;
  if (!GetDiagonal(HT*H, &beta) || !GetDiagonal(H*AT*A*HT, &gamma))
    return false;

  // TODO(mwytock): Deal with zero values in the diagonal scaling beta
  Eigen::VectorXd lambda = alpha*beta.array().square()/gamma.array();

  linear_map::LinearMap B0 = linear_map::Diagonal(beta.cwiseQuotient(gamma));
  linear_map::LinearMap C0 = linear_map::Diagonal(beta.cwiseInverse());
  BlockMatrix B_scale, C_scale;
  for (const std::string& key : H.col_keys()) {
    B_scale(key, key) = B0;
    C_scale(key, key) = C0;
  }

  B_ = H*B_scale*AT;
  C_ = C_scale*HT;
  input_.lambda_vec_ = lambda;
  input_.elementwise_ = true;
  return true;
}

void VectorProx::Init(const ProxOperatorArg& arg) {
  //  if (!InitScalar(arg) && !InitDiagonal(arg))
  if (!InitDiagonal(arg))
    LOG(FATAL) << "Affine transformation is not scalar or diagonal";
  g_ = arg.affine_arg().b;
}

void VectorProx::PreProcessInput(const BlockVector& v) {
  input_.v_ = B_*(v + g_);
}

BlockVector VectorProx::PostProcessOutput() {
  return C_*(output_.x_ - g_);
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
