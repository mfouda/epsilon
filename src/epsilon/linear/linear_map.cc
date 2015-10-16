
#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/sparse_matrix_impl.h"

LinearMap LinearMap::FromDense(DenseMatrix A) {
  return LinearMap(
      std::unique_ptr<LinearMapImpl>(new DenseMatrixImpl(A)));
}

LinearMap LinearMap::FromSparse(SparseMatrix A) {
  return LinearMap(
      std::unique_ptr<LinearMapImpl>(new SparseMatrixImpl(A)));
}
