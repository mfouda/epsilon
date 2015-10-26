
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
  return new DiagonalMatrixImpl(A_.inverse());
}


}  // namespace linear_map
