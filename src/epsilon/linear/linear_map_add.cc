
#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/diagonal_matrix_impl.h"
#include "epsilon/linear/kronecker_product_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"

namespace linear_map {

LinearMap Add(const LinearMapImpl& lhs, const LinearMapImpl& rhs);

LinearMapImpl* Add_DenseMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()+
      static_cast<const DenseMatrixImpl&>(rhs).dense());
}

LinearMapImpl* Add_DenseMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()+
      static_cast<DenseMatrixImpl::DenseMatrix>(
          static_cast<const SparseMatrixImpl&>(rhs).sparse()));
}

LinearMapImpl* Add_DenseMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  DenseMatrixImpl::DenseMatrix A =
      static_cast<const DenseMatrixImpl&>(lhs).dense();
  A.diagonal() +=
      static_cast<const DiagonalMatrixImpl&>(rhs).diagonal().diagonal();
  return new DenseMatrixImpl(A);
}

LinearMapImpl* Add_DenseMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  auto const& S = static_cast<const ScalarMatrixImpl&>(rhs);
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).dense()+
      S.alpha()*DenseMatrixImpl::DenseMatrix::Identity(S.n(), S.n()));
}

LinearMapImpl* Add_DenseMatrix_KroneckerProduct(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  LOG(FATAL) << "Not implemented";
}

LinearMapImpl* Add_SparseMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<DenseMatrixImpl::DenseMatrix>(
          static_cast<const SparseMatrixImpl&>(lhs).sparse())+
      static_cast<const DenseMatrixImpl&>(rhs).dense());
}

LinearMapImpl* Add_SparseMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new SparseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()+
      static_cast<const SparseMatrixImpl&>(rhs).sparse());
}

LinearMapImpl* Add_SparseMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new SparseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()+
      DiagonalSparse(
          static_cast<const DiagonalMatrixImpl&>(rhs).diagonal().diagonal()));
}

LinearMapImpl* Add_SparseMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  auto const& S = static_cast<const ScalarMatrixImpl&>(rhs);
  return new SparseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse()+
      S.alpha()*SparseIdentity(S.n()));
}

LinearMapImpl* Add_SparseMatrix_KroneckerProduct(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  LOG(FATAL) << "Not implemented";
}

LinearMapImpl* Add_DiagonalMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return Add_DenseMatrix_DiagonalMatrix(rhs, lhs);
}

LinearMapImpl* Add_DiagonalMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return Add_SparseMatrix_DiagonalMatrix(rhs, lhs);
}

LinearMapImpl* Add_DiagonalMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  auto const& D = static_cast<const DiagonalMatrixImpl&>(lhs).diagonal();
  auto const& E = static_cast<const DiagonalMatrixImpl&>(rhs).diagonal();
  DiagonalMatrixImpl::DiagonalMatrix F(D.rows());
  F.diagonal().array() = D.diagonal().array()+E.diagonal().array();
  return new DiagonalMatrixImpl(F);
}

LinearMapImpl* Add_DiagonalMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  auto const& D = static_cast<const DiagonalMatrixImpl&>(lhs).diagonal();
  auto const& S = static_cast<const ScalarMatrixImpl&>(rhs);
  DiagonalMatrixImpl::DiagonalMatrix E(D.rows());
  E.diagonal().array() = D.diagonal().array() + S.alpha();
  return new DiagonalMatrixImpl(E);
}

LinearMapImpl* Add_DiagonalMatrix_KroneckerProduct(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  LOG(FATAL) << "Not implemented";
}

LinearMapImpl* Add_ScalarMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return Add_DenseMatrix_ScalarMatrix(rhs, lhs);
}

LinearMapImpl* Add_ScalarMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return Add_SparseMatrix_ScalarMatrix(rhs, lhs);
}

LinearMapImpl* Add_ScalarMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return Add_DiagonalMatrix_ScalarMatrix(rhs, lhs);
}

LinearMapImpl* Add_ScalarMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new ScalarMatrixImpl(
      static_cast<const ScalarMatrixImpl&>(lhs).n(),
      static_cast<const ScalarMatrixImpl&>(lhs).alpha()+
      static_cast<const ScalarMatrixImpl&>(rhs).alpha());
}

LinearMapImpl* Add_ScalarMatrix_KroneckerProduct(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  auto const& K = static_cast<const KroneckerProductImpl&>(rhs);
  auto const& S = static_cast<const ScalarMatrixImpl&>(lhs);

  // kron(A, alpha*I) + beta*I  can be rewritten as
  // kron(A + beta/alpha*I, alpha*I)
  if (K.A().impl().type() == SCALAR_MATRIX) {
    auto const& KS = static_cast<const ScalarMatrixImpl&>(K.A().impl());
    ScalarMatrixImpl S1(K.A().impl().n(), 0);
    ScalarMatrixImpl S2(K.B().impl().n(), S.alpha()/KS.alpha());
    return new KroneckerProductImpl(
        Add(S1, K.A().impl()),
        Add(S2, K.B().impl()));
  } else if (K.B().impl().type() == SCALAR_MATRIX) {
    auto const& KS = static_cast<const ScalarMatrixImpl&>(K.B().impl());
    ScalarMatrixImpl S1(K.A().impl().n(), S.alpha()/KS.alpha());
    ScalarMatrixImpl S2(K.B().impl().n(), 0);
    return new KroneckerProductImpl(
        Add(S1, K.A().impl()),
        Add(S2, K.B().impl()));
  } else {
    LOG(FATAL) << "Not implemented\n"
               << "K:" << K.DebugString() << "\n"
               << "S:" << S.DebugString();
  }
}

LinearMapImpl* Add_KroneckerProduct_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return Add_DenseMatrix_KroneckerProduct(rhs, lhs);
}

LinearMapImpl* Add_KroneckerProduct_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return Add_SparseMatrix_KroneckerProduct(rhs, lhs);
}

LinearMapImpl* Add_KroneckerProduct_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return Add_DiagonalMatrix_KroneckerProduct(rhs, lhs);
}

LinearMapImpl* Add_KroneckerProduct_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return Add_ScalarMatrix_KroneckerProduct(rhs, lhs);
}

LinearMapImpl* Add_KroneckerProduct_KroneckerProduct(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  auto const& K1 = static_cast<const KroneckerProductImpl&>(lhs);
  auto const& K2 = static_cast<const KroneckerProductImpl&>(rhs);

  if (K1.A() == K2.A()) {
    return new KroneckerProductImpl(K1.A(), K1.B() + K2.B());
  } else if (K1.B() == K2.B()) {
    return new KroneckerProductImpl(K1.A() + K2.A(), K1.B());
  } else {
    LOG(FATAL) << "Adding incompatible kronecker products\n"
               << "K1: " << K1.DebugString() << "\n"
               << "K2: " << K2.DebugString();
  }
}

LinearMapImpl* Add_NotImplemented(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  LOG(FATAL) << "Not implemented";
}

LinearMapBinaryOp kAddTable
[NUM_LINEAR_MAP_IMPL_TYPES][NUM_LINEAR_MAP_IMPL_TYPES] = {
  {
    &Add_DenseMatrix_DenseMatrix,
    &Add_DenseMatrix_SparseMatrix,
    &Add_DenseMatrix_DiagonalMatrix,
    &Add_DenseMatrix_ScalarMatrix,
    &Add_DenseMatrix_KroneckerProduct,
    &Add_NotImplemented,
  },
  {
    &Add_SparseMatrix_DenseMatrix,
    &Add_SparseMatrix_SparseMatrix,
    &Add_SparseMatrix_DiagonalMatrix,
    &Add_SparseMatrix_ScalarMatrix,
    &Add_SparseMatrix_KroneckerProduct,
    &Add_NotImplemented,
  },
  {
    &Add_DiagonalMatrix_DenseMatrix,
    &Add_DiagonalMatrix_SparseMatrix,
    &Add_DiagonalMatrix_DiagonalMatrix,
    &Add_DiagonalMatrix_ScalarMatrix,
    &Add_DiagonalMatrix_KroneckerProduct,
    &Add_NotImplemented,
  },
  {
    &Add_ScalarMatrix_DenseMatrix,
    &Add_ScalarMatrix_SparseMatrix,
    &Add_ScalarMatrix_DiagonalMatrix,
    &Add_ScalarMatrix_ScalarMatrix,
    &Add_ScalarMatrix_KroneckerProduct,
    &Add_NotImplemented,
  },
  {
    &Add_KroneckerProduct_DenseMatrix,
    &Add_KroneckerProduct_SparseMatrix,
    &Add_KroneckerProduct_DiagonalMatrix,
    &Add_KroneckerProduct_ScalarMatrix,
    &Add_KroneckerProduct_KroneckerProduct,
    &Add_NotImplemented,
  },
  {
    &Add_NotImplemented,
    &Add_NotImplemented,
    &Add_NotImplemented,
    &Add_NotImplemented,
    &Add_NotImplemented,
    &Add_NotImplemented,
  },
};

LinearMap Add(const LinearMapImpl& lhs, const LinearMapImpl& rhs) {
  return LinearMap((*kAddTable[lhs.type()][rhs.type()])(lhs, rhs));
}

LinearMap operator+(const LinearMap& lhs, const LinearMap& rhs) {
  return Add(lhs.impl(), rhs.impl());
}

}  // namespace linear_map
