#include "epsilon/prox/vector_prox.h"
#include "epsilon/vector/vector_util.h"


void VectorProx::Init(const ProxOperatorArg& arg) {
  InitArgs(arg.affine_arg());
  InitConstraints(arg.affine_constraint());
  InitInput();
}

void VectorProx::InitInput() {
  input_.elementwise_ = elementwise_;
  if (elementwise_) {
    LOG(FATAL) << "elementwise";
  } else {
    input_.lambda_ = alpha_/beta_;
    input_.lambda_ *= input_.lambda_;
    input_.lambda_vec_ = Eigen::VectorXd::Constant(n_, input_.lambda_);
    AT_ = (alpha_/beta_)*AT_;
  }

  // VLOG(2) << "AT: " << AT_.DebugString();
  // VLOG(2) << "lambda: " << lambda_ << ", alpha: " << alpha_;
  // VLOG(2) << "b: " << VectorDebugString(b_);
}

void VectorProx::InitArgs(const AffineOperator& f) {
  // Assumes single argument and single variable
  const BlockMatrix& H = f.A;
  b_ = f.b;

  if (H.row_keys().size() == 1 && H.col_keys().size() == 1) {
    // Single key case
    std::string key = affine::arg_key(0);
    const linear_map::LinearMap& H0 = H(key, *H.col_keys().begin());
    n_ = H0.impl().n();

    if (H0.impl().type() == linear_map::DIAGONAL_MATRIX) {
      elementwise_ = true;
      alpha_vec_ = linear_map::GetDiagonal(H0);
      D_alpha_inv_(key, key) = linear_map::Diagonal(alpha_vec_.cwiseInverse());
    } else if (H0.impl().type() == linear_map::SCALAR_MATRIX) {
      elementwise_ = false;
      alpha_ = linear_map::GetScalar(H0);
      D_alpha_inv_(key, key) = linear_map::Scalar(1/alpha_, n_);
    } else {
      LOG(FATAL) << "Non diagonal scaling";
    }
  } else {
    LOG(FATAL) << "multi arg function";
  }
}

void VectorProx::InitConstraints(const AffineOperator& f) {
  // Handle simple case where A'A is scalar and b=0
  const BlockMatrix& A = f.A;
  const BlockVector& b = f.b;
  CHECK(b.keys().empty());

  AT_ = A.Transpose();
  BlockMatrix ATA = AT_*A;

  bool first = false;
  for (const auto& col_iter : ATA.data()) {
    CHECK(col_iter.second.size() == 1 &&
          col_iter.first == col_iter.second.begin()->first)
        << "Constraint transform A(x), A'A not block diagonal";
    if (first) {
      beta_ = linear_map::GetScalar(col_iter.second.begin()->second);
      first = false;
    } else {
      CHECK_EQ(beta_, linear_map::GetScalar(col_iter.second.begin()->second));
    }
  }
}

// Build input from v
void VectorProx::PreProcessInput(const BlockVector& v) {
  input_.v_ = AT_*v + b_;
}

// Build x from output
BlockVector VectorProx::PostProcessOutput() {
  return D_alpha_inv_*(output_.x_ - b_);
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
