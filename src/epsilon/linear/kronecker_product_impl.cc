#include "epsilon/linear/kronecker_product_impl.h"

namespace linear_map {

LinearMap::DenseMatrix KroneckerProductImpl::AsDense() const {
  DenseMatrix A = A_.impl().AsDense();
  DenseMatrix B = B_.impl().AsDense();
  DenseMatrix C(m(), n());

  for (int i = 0; i < A.rows(); i++) {
    for (int j = 0; j < A.cols(); j++) {
      C.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j)*B;
    }
  }

  return C;
}

LinearMap::DenseMatrix KroneckerProductImpl::Apply(const DenseMatrix& X) const {
  LOG(INFO) << "Apply: " << DebugString();

  // TODO(mwytock): Support this case if needed
  CHECK_EQ(1, X.cols());
  DenseMatrix X_mat = ToMatrix(X, B_.impl().n(), A_.impl().n());
  return ToVector(
      A_.impl().Apply(
          B_.impl().Apply(X_mat).transpose()).transpose());
}

bool KroneckerProductImpl::operator==(const LinearMapImpl& other) const {
  if (other.type() != KRONECKER_PRODUCT ||
      other.m() != m() ||
      other.n() != n())
    return false;

  auto const& K = static_cast<const KroneckerProductImpl&>(other);
  return K.A() == A() && K.B() == B();
}


}  // namespace linear_map
