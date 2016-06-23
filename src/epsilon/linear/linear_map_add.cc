
#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/diagonal_matrix_impl.h"
#include "epsilon/linear/kronecker_product_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"

namespace linear_map {

LinearMapImpl* Add(const LinearMapImpl& lhs, const LinearMapImpl& rhs);

LinearMapImpl* Add_DenseMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).AsDense()+
      static_cast<const DenseMatrixImpl&>(rhs).AsDense());
}

LinearMapImpl* Add_DenseMatrix_SparseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).AsDense()+
      static_cast<DenseMatrixImpl::DenseMatrix>(
          static_cast<const SparseMatrixImpl&>(rhs).sparse()));
}

LinearMapImpl* Add_DenseMatrix_DiagonalMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  DenseMatrixImpl::DenseMatrix A =
      static_cast<const DenseMatrixImpl&>(lhs).AsDense();
  A.diagonal() +=
      static_cast<const DiagonalMatrixImpl&>(rhs).diagonal().diagonal();
  return new DenseMatrixImpl(A);
}

LinearMapImpl* Add_DenseMatrix_ScalarMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  auto const& S = static_cast<const ScalarMatrixImpl&>(rhs);
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).AsDense()+
      S.alpha()*DenseMatrixImpl::DenseMatrix::Identity(S.n(), S.n()));
}

LinearMapImpl* Add_DenseMatrix_KroneckerProduct(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<const DenseMatrixImpl&>(lhs).AsDense()+rhs.AsDense());
}

LinearMapImpl* Add_SparseMatrix_DenseMatrix(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  return new DenseMatrixImpl(
      static_cast<DenseMatrixImpl::DenseMatrix>(
          static_cast<const SparseMatrixImpl&>(lhs).sparse())+
      static_cast<const DenseMatrixImpl&>(rhs).AsDense());
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
  return new SparseMatrixImpl(
      static_cast<const SparseMatrixImpl&>(lhs).sparse() +
      static_cast<const KroneckerProductImpl&>(rhs).AsSparse());
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
  return new SparseMatrixImpl(
      DiagonalSparse(
          static_cast<const DiagonalMatrixImpl&>(lhs).diagonal().diagonal()) +
      static_cast<const KroneckerProductImpl&>(rhs).AsSparse());
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
  auto const& S = static_cast<const ScalarMatrixImpl&>(lhs);
  auto const& K = static_cast<const KroneckerProductImpl&>(rhs);

  // kron(A, alpha*I) + beta*I  can be rewritten as
  // kron(A + beta/alpha*I, alpha*I)
  if (K.A().impl().type() == SCALAR_MATRIX) {
    auto const& KS = static_cast<const ScalarMatrixImpl&>(K.A().impl());
    LinearMap S1 = linear_map::Scalar(0, K.A().impl().n());
    LinearMap S2 = linear_map::Scalar(S.alpha()/KS.alpha(), K.B().impl().n());
    return new KroneckerProductImpl(S1 + K.A(), S2 + K.B());
  } else if (K.B().impl().type() == SCALAR_MATRIX) {
    auto const& KS = static_cast<const ScalarMatrixImpl&>(K.B().impl());
    LinearMap S1 = linear_map::Scalar(S.alpha()/KS.alpha(), K.A().impl().n());
    LinearMap S2 = linear_map::Scalar(0, K.B().impl().n());
    return new KroneckerProductImpl(S1 + K.A(), S2 + K.B());
  }
  return new SparseMatrixImpl(S.AsSparse() + K.AsSparse());
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
    return new SparseMatrixImpl(K1.AsSparse() + K2.AsSparse());
  }
}

LinearMapImpl* Add_NotImplemented(
    const LinearMapImpl& lhs,
    const LinearMapImpl& rhs) {
  LOG(FATAL) << "Not implemented";
}

LinearMapBinaryOp kAddTable
[NUM_IMPL_TYPES][NUM_IMPL_TYPES] = {
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

LinearMapImpl* Add(const LinearMapImpl& lhs, const LinearMapImpl& rhs) {
  VLOG(2) << "linear_map_add " << lhs.type() << " " << rhs.type();
  return (*kAddTable[lhs.type()][rhs.type()])(lhs, rhs);
}

LinearMap operator+(const LinearMap& lhs, const LinearMap& rhs) {
  return LinearMap(Add(lhs.impl(), rhs.impl()));
}

}  // namespace linear_map
