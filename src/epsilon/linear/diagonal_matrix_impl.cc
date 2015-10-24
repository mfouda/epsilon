
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

}  // namespace linear_map
