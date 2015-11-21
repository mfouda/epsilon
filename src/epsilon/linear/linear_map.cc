
#include <unordered_map>

#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/diagonal_matrix_impl.h"
#include "epsilon/linear/kronecker_product_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"
#include "epsilon/vector/vector_file.h"

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

LinearMap KroneckerProduct(const ::LinearMap& proto) {
  CHECK_EQ(2, proto.arg_size());
  return LinearMap(new KroneckerProductImpl(
      BuildLinearMap(proto.arg(0)),
      BuildLinearMap(proto.arg(1))));
}

LinearMap DenseMatrix(const ::LinearMap& proto) {
  return LinearMap(new DenseMatrixImpl(ReadMatrixData(proto.constant())));
}

LinearMap DiagonalMatrix(const ::LinearMap& proto) {
  return LinearMap(new DiagonalMatrixImpl(
      ToVector(ReadMatrixData(proto.constant())).asDiagonal()));
}

LinearMap Scalar(const ::LinearMap& proto) {
  return LinearMap(new ScalarMatrixImpl(proto.n(), proto.scalar()));
}

LinearMap Transpose(const ::LinearMap& proto) {
  CHECK_EQ(1, proto.arg_size());
  return BuildLinearMap(proto.arg(0)).Transpose();
}

LinearMap SparseMatrix(const ::LinearMap& proto) {
  return LinearMap(new SparseMatrixImpl(ReadSparseMatrixData(proto.constant())));
}

typedef LinearMap(*LinearMapFunction)(const ::LinearMap&);

std::unordered_map<int, LinearMapFunction> kLinearMapFunctions = {
  {::LinearMap::DENSE_MATRIX, &DenseMatrix},
  {::LinearMap::DIAGONAL_MATRIX, &DiagonalMatrix},
  {::LinearMap::KRONECKER_PRODUCT, &KroneckerProduct},
  {::LinearMap::SCALAR, &Scalar},
  {::LinearMap::SPARSE_MATRIX, &SparseMatrix},
  {::LinearMap::TRANSPOSE, &Transpose},
};

LinearMap BuildLinearMap(const ::LinearMap& linear_map) {
  auto iter = kLinearMapFunctions.find(linear_map.linear_map_type());
  if (iter == kLinearMapFunctions.end()) {
    LOG(FATAL) << "No linear map function for "
               << ::LinearMap::Type_Name(linear_map.linear_map_type());
  }
  return iter->second(linear_map);
}

LinearMap Identity(int n) {
  return LinearMap(new ScalarMatrixImpl(n, 1));
}

LinearMap Diagonal(const Eigen::VectorXd& a) {
  return LinearMap(new DiagonalMatrixImpl(a.asDiagonal()));
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

}  // namespace linear_map
