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

LinearMap::DenseVector KroneckerProductImpl::Apply(const DenseVector& x) const {
  LinearMap X(new DenseMatrixImpl(ToMatrix(x, B_.impl().n(), A_.impl().n())));
  LinearMap Y = (A_*(B_*X).Transpose()).Transpose();
  return ToVector(Y.impl().AsDense());
}

}  // namespace linear_map
