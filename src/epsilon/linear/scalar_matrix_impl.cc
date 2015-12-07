#include "epsilon/linear/scalar_matrix_impl.h"

namespace linear_map {

bool ScalarMatrixImpl::operator==(const LinearMapImpl& other) const {
  if (other.type() != SCALAR_MATRIX ||
      other.m() != m() ||
      other.n() != n())
    return false;
  return static_cast<const ScalarMatrixImpl&>(other).alpha() == alpha_;
}

LinearMap::SparseMatrix ScalarMatrixImpl::AsSparse() const {
  SparseMatrix A(n(), n());
  A.setIdentity();
  A *= alpha_;
  return A;
}

}  // namespace linear_map
