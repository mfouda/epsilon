#include "epsilon/linear/dense_matrix_impl.h"

LinearMapImpl* DenseMatrixImpl::Inverse() const {
  // TODO(mwytock): LLT method may be faster?
  return new DenseMatrixImpl(A_.inverse());
}
