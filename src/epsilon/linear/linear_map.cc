
#include <unordered_map>

#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/diagonal_matrix_impl.h"
#include "epsilon/linear/kronecker_product_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"

namespace linear_map {

LinearMap::LinearMap() : impl_(new ScalarMatrixImpl(0, 0)) {}

LinearMap& LinearMap::operator+=(const LinearMap& rhs) {
  *this = *this+rhs;
  return *this;
}

LinearMap& LinearMap::operator*=(const LinearMap& rhs) {
  *this = *this*rhs;
  return *this;
}

bool operator==(const LinearMap& lhs, const LinearMap& rhs) {
  // TODO(mwytock): If we had caching of LinearMaps we could just compare
  // pointers here?
  return lhs.impl() == rhs.impl();
}

LinearMap operator*(double alpha, const LinearMap& A) {
  return LinearMap(new ScalarMatrixImpl(A.impl().m(), alpha))*A;
}

LinearMap operator*(const LinearMap& A, double alpha) {
  return alpha*A;
}

LinearMapImpl* BuildLinearMapImpl(
  const ::LinearMap& linear_map, const DataMap& data_map);

LinearMapImpl* KroneckerProduct(
    const ::LinearMap& proto, const DataMap& data_map) {
  CHECK_EQ(2, proto.arg_size());
  return new KroneckerProductImpl(
      BuildLinearMap(proto.arg(0), data_map),
      BuildLinearMap(proto.arg(1), data_map));
}

LinearMapImpl* DenseMatrix(
    const ::LinearMap& proto, const DataMap& data_map) {
  return new DenseMatrixImpl(
      BuildMatrix(proto.constant(), data_map));
}

LinearMapImpl* DiagonalMatrix(
    const ::LinearMap& proto, const DataMap& data_map) {
  return new DiagonalMatrixImpl(
      ToVector(BuildMatrix(proto.constant(), data_map)).asDiagonal());
}

LinearMapImpl* Scalar(
    const ::LinearMap& proto, const DataMap& data_map) {
  return new ScalarMatrixImpl(proto.n(), proto.scalar());
}

LinearMapImpl* Transpose(
    const ::LinearMap& proto, const DataMap& data_map) {
  CHECK_EQ(1, proto.arg_size());
  std::unique_ptr<LinearMapImpl> A(BuildLinearMapImpl(proto.arg(0), data_map));
  return A->Transpose();
}

LinearMapImpl* SparseMatrix(
    const ::LinearMap& proto, const DataMap& data_map) {
  return new SparseMatrixImpl(
    BuildSparseMatrix(proto.constant(), data_map));
}

typedef LinearMapImpl*(*LinearMapFunction)(
    const ::LinearMap& linear_map, const DataMap& data_map);

std::unordered_map<int, LinearMapFunction> kLinearMapFunctions = {
  {::LinearMap::DENSE_MATRIX, &DenseMatrix},
  {::LinearMap::DIAGONAL_MATRIX, &DiagonalMatrix},
  {::LinearMap::KRONECKER_PRODUCT, &KroneckerProduct},
  {::LinearMap::SCALAR, &Scalar},
  {::LinearMap::SPARSE_MATRIX, &SparseMatrix},
  {::LinearMap::TRANSPOSE, &Transpose},
};

LinearMapImpl* BuildLinearMapImpl(
  const ::LinearMap& linear_map, const DataMap& data_map) {
  auto iter = kLinearMapFunctions.find(linear_map.linear_map_type());
  if (iter == kLinearMapFunctions.end()) {
    LOG(FATAL) << "No linear map function for "
               << ::LinearMap::Type_Name(linear_map.linear_map_type());
  }
  return iter->second(linear_map, data_map);
}

LinearMap BuildLinearMap(const ::LinearMap& linear_map, const DataMap& data_map) {
  return LinearMap(BuildLinearMapImpl(linear_map, data_map));
}

LinearMap Identity(int n) {
  return LinearMap(new ScalarMatrixImpl(n, 1));
}

LinearMap Diagonal(const Eigen::VectorXd& a) {
  return LinearMap(new DiagonalMatrixImpl(a.asDiagonal()));
}

LinearMap Scalar(double alpha, int n) {
  return LinearMap(new ScalarMatrixImpl(n, alpha));
}

Eigen::VectorXd GetDiagonal(const LinearMap& linear_map) {
  const LinearMapImpl& impl = linear_map.impl();
  if (impl.type() == SCALAR_MATRIX) {
    const ScalarMatrixImpl& S = static_cast<const ScalarMatrixImpl&>(impl);
    return Eigen::VectorXd::Constant(S.n(), S.alpha());
  } else if (impl.type() == DIAGONAL_MATRIX) {
    const DiagonalMatrixImpl& D = static_cast<const DiagonalMatrixImpl&>(impl);
    return D.diagonal().diagonal();
  } else {
    LOG(FATAL) << "Non-diagonal linear map " << impl.type();
  }
}

double GetScalar(const LinearMap& linear_map) {
  const LinearMapImpl& impl = linear_map.impl();
  if (impl.type() == SCALAR_MATRIX) {
    const ScalarMatrixImpl& S = static_cast<const ScalarMatrixImpl&>(impl);
    return S.alpha();
  } else {
    LOG(FATAL) << "Non-scalar matrix " << impl.type();
  }
}

ImplType ComputeType(OpType type, ImplType A, ImplType B) {
  // Basic promotion
  if (A <= SCALAR_MATRIX && B <= SCALAR_MATRIX) {
    return A < B ? A : B;
  }
  // TODO(mwytock): Improve this with more complete type information, e.g. for
  // kronecker product.
  return DENSE_MATRIX;
}

uint64_t Nonzeros(ImplType type, int m, int n) {
  switch (type) {
    case DENSE_MATRIX:
    case SPARSE_MATRIX:
      return m*n;
    case DIAGONAL_MATRIX:
      CHECK_EQ(m, n);
      return n;
    case SCALAR_MATRIX:
      return 1;
    default:
      LOG(FATAL) << "Not implemented";
  }
}


}  // namespace linear_map
