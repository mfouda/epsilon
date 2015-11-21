
#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/diagonal_matrix_impl.h"
#include "epsilon/linear/kronecker_product_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"

namespace linear_map {

LinearMap Multiply(const LinearMapImpl& lhs, const LinearMapImpl& rhs);

LinearMapImpl* Multiply_DenseMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()*
      static_cast<const DenseMatrixImpl&>(rhs).dense());
}

LinearMapImpl* Multiply_DenseMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()*
      static_cast<const SparseMatrixImpl&>(rhs).sparse());
}

LinearMapImpl* Multiply_DenseMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()*
      static_cast<const DiagonalMatrixImpl&>(rhs).diagonal());
}

LinearMapImpl* Multiply_DenseMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()*
      static_cast<const ScalarMatrixImpl&>(rhs).alpha());
}

LinearMapImpl* Multiply_DenseMatrix_KroneckerProduct(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()*
      rhs.AsDense());
}

LinearMapImpl* Multiply_SparseMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()*
      static_cast<const DenseMatrixImpl&>(rhs).dense());
}

LinearMapImpl* Multiply_SparseMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new SparseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()*
      static_cast<const SparseMatrixImpl&>(rhs).sparse());
}

LinearMapImpl* Multiply_SparseMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new SparseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()*
      static_cast<const DiagonalMatrixImpl&>(rhs).diagonal());
}

LinearMapImpl* Multiply_SparseMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new SparseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()*
      static_cast<const ScalarMatrixImpl&>(rhs).alpha());
}

LinearMapImpl* Multiply_SparseMatrix_KroneckerProduct(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new SparseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()*
      static_cast<const KroneckerProductImpl&>(rhs).AsSparse());
}

LinearMapImpl* Multiply_DiagonalMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const DiagonalMatrixImpl&>(lhs).diagonal()*
      static_cast<const DenseMatrixImpl&>(rhs).dense());
}

LinearMapImpl* Multiply_DiagonalMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new SparseMatrixImpl(
      static_cast<const DiagonalMatrixImpl&>(lhs).diagonal()*
      static_cast<const SparseMatrixImpl&>(rhs).sparse());
}

LinearMapImpl* Multiply_DiagonalMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  const DiagonalMatrixImpl::DiagonalMatrix& A =
      static_cast<const DiagonalMatrixImpl&>(lhs).diagonal();
  const DiagonalMatrixImpl::DiagonalMatrix& B =
      static_cast<const DiagonalMatrixImpl&>(rhs).diagonal();
  DiagonalMatrixImpl::DiagonalMatrix C(A.rows());
  C.diagonal().array() = A.diagonal().array()*B.diagonal().array();
  return new DiagonalMatrixImpl(C);
}

LinearMapImpl* Multiply_DiagonalMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DiagonalMatrixImpl(
      static_cast<const DiagonalMatrixImpl&>(lhs).diagonal()*
      static_cast<const ScalarMatrixImpl&>(rhs).alpha());
}

LinearMapImpl* Multiply_DiagonalMatrix_KroneckerProduct(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  LOG(FATAL) << "Not implemented";
}

LinearMapImpl* Multiply_ScalarMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const ScalarMatrixImpl&>(lhs).alpha()*
      static_cast<const DenseMatrixImpl&>(rhs).dense());
}

LinearMapImpl* Multiply_ScalarMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new SparseMatrixImpl(
      static_cast<const ScalarMatrixImpl&>(lhs).alpha()*
      static_cast<const SparseMatrixImpl&>(rhs).sparse());
}

LinearMapImpl* Multiply_ScalarMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DiagonalMatrixImpl(
      static_cast<const ScalarMatrixImpl&>(lhs).alpha()*
      static_cast<const DiagonalMatrixImpl&>(rhs).diagonal());
}

LinearMapImpl* Multiply_ScalarMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new ScalarMatrixImpl(
      static_cast<const ScalarMatrixImpl&>(lhs).n(),
      static_cast<const ScalarMatrixImpl&>(lhs).alpha()*
      static_cast<const ScalarMatrixImpl&>(rhs).alpha());
}

LinearMapImpl* Multiply_ScalarMatrix_KroneckerProduct(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  auto const& S = static_cast<const ScalarMatrixImpl&>(lhs);
  auto const& K = static_cast<const KroneckerProductImpl&>(rhs);
  ScalarMatrixImpl S1(K.A().impl().m(), S.alpha());
  ScalarMatrixImpl S2(K.B().impl().m(), 1);
  return new KroneckerProductImpl(
      Multiply(S1, K.A().impl()),
      Multiply(S2, K.B().impl()));
}

LinearMapImpl* Multiply_KroneckerProduct_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      lhs.AsDense()*
      static_cast<const DenseMatrixImpl&>(rhs).dense());
}

LinearMapImpl* Multiply_KroneckerProduct_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new SparseMatrixImpl(
      static_cast<const KroneckerProductImpl&>(lhs).AsSparse()*
      static_cast<const SparseMatrixImpl&>(rhs).sparse());
}

LinearMapImpl* Multiply_KroneckerProduct_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  LOG(FATAL) << "Not implemented";
}

LinearMapImpl* Multiply_KroneckerProduct_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return Multiply_ScalarMatrix_KroneckerProduct(rhs, lhs);
}

LinearMapImpl* Multiply_KroneckerProduct_KroneckerProduct(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  const KroneckerProductImpl& C = static_cast<const KroneckerProductImpl&>(lhs);
  const KroneckerProductImpl& D = static_cast<const KroneckerProductImpl&>(rhs);
  if (C.A().impl().n() == D.A().impl().m() &&
      C.B().impl().n() == D.B().impl().m()) {
  return new KroneckerProductImpl(
      Multiply(C.A().impl(), D.A().impl()),
      Multiply(C.B().impl(), D.B().impl()));
  }

  LOG(FATAL) << "Not implemented: "
             << "C: " << C.DebugString() << "\n"
             << "D: " << D.DebugString();

}

LinearMapImpl* Multiply_NotImplemented(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  LOG(FATAL) << "Not implemented";
}

LinearMapBinaryOp kMultiplyTable
[NUM_LINEAR_MAP_IMPL_TYPES][NUM_LINEAR_MAP_IMPL_TYPES] = {
  {
    &Multiply_DenseMatrix_DenseMatrix,
    &Multiply_DenseMatrix_SparseMatrix,
    &Multiply_DenseMatrix_DiagonalMatrix,
    &Multiply_DenseMatrix_ScalarMatrix,
    &Multiply_DenseMatrix_KroneckerProduct,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_SparseMatrix_DenseMatrix,
    &Multiply_SparseMatrix_SparseMatrix,
    &Multiply_SparseMatrix_DiagonalMatrix,
    &Multiply_SparseMatrix_ScalarMatrix,
    &Multiply_SparseMatrix_KroneckerProduct,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_DiagonalMatrix_DenseMatrix,
    &Multiply_DiagonalMatrix_SparseMatrix,
    &Multiply_DiagonalMatrix_DiagonalMatrix,
    &Multiply_DiagonalMatrix_ScalarMatrix,
    &Multiply_DiagonalMatrix_KroneckerProduct,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_ScalarMatrix_DenseMatrix,
    &Multiply_ScalarMatrix_SparseMatrix,
    &Multiply_ScalarMatrix_DiagonalMatrix,
    &Multiply_ScalarMatrix_ScalarMatrix,
    &Multiply_ScalarMatrix_KroneckerProduct,
    &Multiply_NotImplemented,
  },
  {
    &Multiply_KroneckerProduct_DenseMatrix,
    &Multiply_KroneckerProduct_SparseMatrix,
    &Multiply_KroneckerProduct_DiagonalMatrix,
    &Multiply_KroneckerProduct_ScalarMatrix,
    &Multiply_KroneckerProduct_KroneckerProduct,
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

LinearMap Multiply(const LinearMapImpl& lhs, const LinearMapImpl& rhs) {
  CHECK_EQ(lhs.n(), rhs.m());
  return LinearMap((*kMultiplyTable[lhs.type()][rhs.type()])(lhs, rhs));
}

LinearMap operator*(const LinearMap& lhs, const LinearMap& rhs) {
  return LinearMap(Multiply(lhs.impl(), rhs.impl()));
}

}  // namespace linear_map
