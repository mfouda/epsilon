
#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/sparse_matrix_impl.h"

std::unique_ptr<LinearMapImpl> Multiply_DenseMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()*
      static_cast<const DenseMatrixImpl&>(rhs).dense()));
}

std::unique_ptr<LinearMapImpl> Multiply_DenseMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()*
      static_cast<const SparseMatrixImpl&>(rhs).sparse()));
}

std::unique_ptr<LinearMapImpl> Multiply_SparseMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new DenseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()*
      static_cast<const DenseMatrixImpl&>(rhs).dense()));
}

std::unique_ptr<LinearMapImpl> Multiply_SparseMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new SparseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()*
      static_cast<const SparseMatrixImpl&>(rhs).sparse()));
}

std::unique_ptr<LinearMapImpl> Multiply_NotImplemented(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  LOG(FATAL) << "Not implemented";
}

typedef std::unique_ptr<LinearMapImpl> (*MultiplyLinearMap)(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs);

MultiplyLinearMap kMultiplyTable
[NUM_LINEAR_MAP_IMPL_TYPES][NUM_LINEAR_MAP_IMPL_TYPES] = {
  {
    &Multiply_DenseMatrix_DenseMatrix,
    &Multiply_DenseMatrix_SparseMatrix,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_SparseMatrix_DenseMatrix,
    &Multiply_SparseMatrix_SparseMatrix,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
  },
};

LinearMap operator*(const LinearMap& lhs, const LinearMap& rhs) {
  return LinearMap((*kMultiplyTable[lhs.impl().type()][rhs.impl().type()])(
      lhs.impl(), rhs.impl()));
}
