
#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/diagonal_matrix_impl.h"
#include "epsilon/linear/kronecker_product_impl.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"

LinearMap::LinearMap() : impl_(new ScalarMatrixImpl(0, 0)) {}

LinearMap& LinearMap::operator+=(const LinearMap& rhs) {
  *this = *this+rhs;
  return *this;
}

LinearMap& LinearMap::operator*=(const LinearMap& rhs) {
  *this = *this*rhs;
  return *this;
}

LinearMap operator*(double alpha, const LinearMap& A) {
  return LinearMap(new ScalarMatrixImpl(A.impl().m(), alpha))*A;
}

LinearMap operator*(const LinearMap& A, double alpha) {
  return alpha*A;
}

namespace linear_map {

LinearMap Dense(const LinearMap::DenseMatrix& A) {
  return LinearMap(new DenseMatrixImpl(A));
}

LinearMap Diagonal(const LinearMap::DenseVector& a) {
  DiagonalMatrixImpl::DiagonalMatrix D(a.rows());
  D.diagonal() = a;
  return LinearMap(new DiagonalMatrixImpl(D));
}

LinearMap Sparse(const LinearMap::SparseMatrix& A) {
  return LinearMap(new SparseMatrixImpl(A));
}

LinearMap Identity(int n) {
  return Scale(n, 1);
}

LinearMap Negate(int n) {
  return Scale(n, -1);
}

LinearMap OneHot(int i, int n) {
  LinearMap::SparseMatrix P(1, n);
  P.coeffRef(0,i) = 1;
  P.makeCompressed();
  return Sparse(P);
}

LinearMap Scale(int n, double alpha) {
  return LinearMap(new ScalarMatrixImpl(n, alpha));
}

LinearMap Sum(int n) {
  return Dense(LinearMap::DenseMatrix::Constant(1, n, 1));
}

LinearMap MatrixTranspose(int m, int n) {
  SparseMatrixImpl::SparseMatrix T(m*n,m*n);
  {
    std::vector<Eigen::Triplet<double> > coeffs;
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        coeffs.push_back(Eigen::Triplet<double>(j*m + i, i*n + j, 1));
      }
    }
    T.setFromTriplets(coeffs.begin(), coeffs.end());
  }
  return LinearMap(new SparseMatrixImpl(T));
}

LinearMap Index(int start, int stop, int step, int n) {
  const int m  = stop - start;
  CHECK(m % step == 0);

  SparseMatrixImpl::SparseMatrix P(m,n);
  {
    std::vector<Eigen::Triplet<double> > coeffs;
    for (int i = 0; i < n; i++) {
      coeffs.push_back(Eigen::Triplet<double>(i, start + i*step, 1));
    }
    P.setFromTriplets(coeffs.begin(), coeffs.end());
  }
  return LinearMap(new SparseMatrixImpl(P));
}

LinearMap KroneckerProduct(const LinearMap& A, const LinearMap& B) {
  return LinearMap(new KroneckerProductImpl(A, B));
}

// vec(AX) = kron(I,A)vec(X)
LinearMap MatrixProductLeft(const LinearMap& A, int n) {
  return KroneckerProduct(Identity(n), A);
}

// vec(XA) = kron(A',I)vec(X)
LinearMap MatrixProductRight(const LinearMap& A, int m) {
  return KroneckerProduct(A.Transpose(), Identity(m));
}

}  // namespace linear_map
