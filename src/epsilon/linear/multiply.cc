
#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/diagonal_matrix_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
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

std::unique_ptr<LinearMapImpl> Multiply_DenseMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()*
      static_cast<const DiagonalMatrixImpl&>(rhs).diagonal()));
}

std::unique_ptr<LinearMapImpl> Multiply_DenseMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()*
      static_cast<const ScalarMatrixImpl&>(rhs).alpha()));
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

std::unique_ptr<LinearMapImpl> Multiply_SparseMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new SparseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()*
      static_cast<const DiagonalMatrixImpl&>(rhs).diagonal()));
}

std::unique_ptr<LinearMapImpl> Multiply_SparseMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new SparseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()*
      static_cast<const ScalarMatrixImpl&>(rhs).alpha()));
}

std::unique_ptr<LinearMapImpl> Multiply_DiagonalMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new DenseMatrixImpl(
      static_cast<const DiagonalMatrixImpl&>(lhs).diagonal()*
      static_cast<const DenseMatrixImpl&>(rhs).dense()));
}

std::unique_ptr<LinearMapImpl> Multiply_DiagonalMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new SparseMatrixImpl(
      static_cast<const DiagonalMatrixImpl&>(lhs).diagonal()*
      static_cast<const SparseMatrixImpl&>(rhs).sparse()));
}

std::unique_ptr<LinearMapImpl> Multiply_DiagonalMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  const DiagonalMatrixImpl::DiagonalMatrix& A =
      static_cast<const DiagonalMatrixImpl&>(lhs).diagonal();
  const DiagonalMatrixImpl::DiagonalMatrix& B =
      static_cast<const DiagonalMatrixImpl&>(rhs).diagonal();
  DiagonalMatrixImpl::DiagonalMatrix C(A.rows());
  C.diagonal().array() = A.diagonal().array()*B.diagonal().array();
  return std::unique_ptr<LinearMapImpl>(new DiagonalMatrixImpl(C));
}

std::unique_ptr<LinearMapImpl> Multiply_DiagonalMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new DiagonalMatrixImpl(
      static_cast<const DiagonalMatrixImpl&>(lhs).diagonal()*
      static_cast<const ScalarMatrixImpl&>(rhs).alpha()));
}

std::unique_ptr<LinearMapImpl> Multiply_ScalarMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new DenseMatrixImpl(
      static_cast<const ScalarMatrixImpl&>(lhs).alpha()*
      static_cast<const DenseMatrixImpl&>(rhs).dense()));
}

std::unique_ptr<LinearMapImpl> Multiply_ScalarMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new SparseMatrixImpl(
      static_cast<const ScalarMatrixImpl&>(lhs).alpha()*
      static_cast<const SparseMatrixImpl&>(rhs).sparse()));
}

std::unique_ptr<LinearMapImpl> Multiply_ScalarMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new DiagonalMatrixImpl(
      static_cast<const ScalarMatrixImpl&>(lhs).alpha()*
      static_cast<const DiagonalMatrixImpl&>(rhs).diagonal()));
}

std::unique_ptr<LinearMapImpl> Multiply_ScalarMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return std::unique_ptr<LinearMapImpl>(new ScalarMatrixImpl(
      static_cast<const ScalarMatrixImpl&>(lhs).n(),
      static_cast<const ScalarMatrixImpl&>(lhs).alpha()*
      static_cast<const ScalarMatrixImpl&>(rhs).alpha()));
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
    &Multiply_DenseMatrix_DiagonalMatrix,
    &Multiply_DenseMatrix_ScalarMatrix,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_SparseMatrix_DenseMatrix,
    &Multiply_SparseMatrix_SparseMatrix,
    &Multiply_SparseMatrix_DiagonalMatrix,
    &Multiply_SparseMatrix_ScalarMatrix,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_DiagonalMatrix_DenseMatrix,
    &Multiply_DiagonalMatrix_SparseMatrix,
    &Multiply_DiagonalMatrix_DiagonalMatrix,
    &Multiply_DiagonalMatrix_ScalarMatrix,
    &Multiply_NotImplemented,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_ScalarMatrix_DenseMatrix,
    &Multiply_ScalarMatrix_SparseMatrix,
    &Multiply_ScalarMatrix_DiagonalMatrix,
    &Multiply_ScalarMatrix_ScalarMatrix,
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
