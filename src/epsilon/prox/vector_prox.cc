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
  CHECK_GE(input_.lambda_, 0);
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

  // Deal with zero values in the diagonal scaling beta/gamma
  const int n = beta.rows();
  Eigen::VectorXd lambda(n);
  Eigen::VectorXd delta = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < n; i++) {
    if (gamma(i)) {
      lambda(i) = alpha*beta(i)*beta(i)/gamma(i);
    } else {
      lambda(i) = 0;
      beta(i) = 1;
      gamma(i) = 1;
      delta(i) = 1;
    }
  }

  // This scaling is somewhat nasty but there is no more concise way to specify
  // the elementwise function.
  linear_map::LinearMap B0 = linear_map::Diagonal(beta.cwiseQuotient(gamma));
  linear_map::LinearMap C0 = linear_map::Diagonal(beta.cwiseInverse());
  linear_map::LinearMap D0 = linear_map::Diagonal(delta);
  BlockMatrix B_scale, C_scale, D_scale;
  for (const std::string& key : H.col_keys()) {
    B_scale(key, key) = B0;
    C_scale(key, key) = C0;
    D_scale(key, key) = D0;
  }
  B_ = H*B_scale*AT;
  C_ = C_scale*HT;
  D_ = (AT*A).Inverse()*D_scale*AT;

  input_.lambda_vec_ = lambda;
  input_.elementwise_ = true;
  return true;
}

void VectorProx::Init(const ProxOperatorArg& arg) {
  if (!InitScalar(arg) && !InitDiagonal(arg))
    LOG(FATAL) << "Affine transformation is not scalar or diagonal";
  g_ = arg.affine_arg().b;

  VLOG(2) << "B: " << B_.DebugString();
  VLOG(2) << "C: " << C_.DebugString();
  if (input_.elementwise_)
    VLOG(2) << "lambda: " << VectorDebugString(input_.lambda_vec_);
  else
    VLOG(2) << "lambda: " << input_.lambda_;

  InitAxis(arg);
}

void VectorProx::InitAxis(const ProxOperatorArg& arg) {
  prox_function_ = arg.prox_function();
  input_.prox_function_ = arg.prox_function();
  output_.prox_function_ = arg.prox_function();
}

void VectorProx::PreProcessInput(const BlockVector& v) {
  input_.v_ = B_*v + g_;
}

BlockVector VectorProx::PostProcessOutput(const BlockVector& v) {
  return C_*(output_.x_ - g_) + D_*v;
}

BlockVector VectorProx::Apply(const BlockVector& v) {
  PreProcessInput(v);

  LOG(INFO) << prox_function_.DebugString();

  if (prox_function_.has_axis()) {
    const int n = prox_function_.arg_size_size();

    // Set up inputs
    input_.V_.resize(n);
    output_.X_.resize(n);
    if (prox_function_.has_axis()) {
      for (int i = 0; i < n; i++) {
        const Size& dims = prox_function_.arg_size(i);
        input_.V_[i] = ToMatrix(
            input_.v_(affine::arg_key(i)), dims.dim(0), dims.dim(1));
        output_.X_[i] = Eigen::MatrixXd(dims.dim(0), dims.dim(1));
      }
    }

    // Iterate over the other dimension
    const Size& dims = prox_function_.arg_size(0);
    const int k = dims.dim(1 - prox_function_.axis());
    for (int i = 0; i < k; i++) {
      input_.axis_iter_ = i;
      output_.axis_iter_ = i;
      ApplyVector(input_, &output_);
    }

    // Copy outputs
    for (int i = 0; i < n; i++) {
      output_.x_(affine::arg_key(i)) = ToVector(output_.X_[i]);
    }
  } else {
    ApplyVector(input_, &output_);
  }

  return PostProcessOutput(v);
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

Eigen::VectorXd VectorProxInput::value_vec(int i) const {
  if (prox_function_.has_axis()) {
    if (prox_function_.axis() == 0) {
      return V_[i].col(axis_iter_);
    } else {
      return V_[i].row(axis_iter_);
    }
  } else {
    return v_(affine::arg_key(i));
  }
}

void VectorProxInput::set_lambda(double lambda) {
  if(elementwise_) {
    lambda_vec_ = Eigen::VectorXd::Constant(lambda_vec_.rows(), lambda);
  } else {
    lambda_ = lambda;
  }
}

void VectorProxOutput::set_value(int i, double x) {
  set_value(i, Eigen::VectorXd::Constant(1, x));
}

void VectorProxOutput::set_value(int i, const Eigen::VectorXd& x) {
  if (prox_function_.has_axis()) {
    if (prox_function_.axis() == 0) {
      X_[i].col(axis_iter_) = x;
    } else {
      X_[i].row(axis_iter_) = x;
    }
  } else {
    x_(affine::arg_key(i)) = x;
  }
}

double VectorProxOutput::value(int i) const {
  const Eigen::VectorXd& val = value_vec(i);
  CHECK_EQ(1, val.size());
  return val(0);
}

const Eigen::VectorXd& VectorProxOutput::value_vec(int i) const {
  return x_(affine::arg_key(i));
}
