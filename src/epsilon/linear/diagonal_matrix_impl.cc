
#include "epsilon/linear/diagonal_matrix_impl.h"

namespace linear_map {

bool DiagonalMatrixImpl::operator==(const LinearMapImpl& other) const {
  if (other.type() != DIAGONAL_MATRIX ||
      other.m() != m() ||
      other.n() != n())
    return false;
  return (static_cast<const DiagonalMatrixImpl&>(other).diagonal().diagonal() ==
          diagonal().diagonal());
}

LinearMapImpl* DiagonalMatrixImpl::Inverse() const {
  const int n = A_.rows();
  const Eigen::VectorXd& a = A_.diagonal();
  Eigen::VectorXd ainv(n);
  for (int i = 0; i < n; i++) {
    ainv(i) = a(i) ? 1/a(i) : 0;
  }
  return new DiagonalMatrixImpl(ainv.asDiagonal());
}


}  // namespace linear_map
