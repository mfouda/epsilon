
#include <unordered_map>

#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/kronecker_product_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
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

LinearMap Scalar(const ::LinearMap& proto) {
  return LinearMap(new ScalarMatrixImpl(proto.n(), proto.scalar()));
}

LinearMap Transpose(const ::LinearMap& proto) {
  CHECK_EQ(1, proto.arg_size());
  return BuildLinearMap(proto.arg(0)).Transpose();
}

typedef LinearMap(*LinearMapFunction)(const ::LinearMap&);

std::unordered_map<int, LinearMapFunction> kLinearMapFunctions = {
  {::LinearMap::DENSE_MATRIX, &DenseMatrix},
  {::LinearMap::KRONECKER_PRODUCT, &KroneckerProduct},
  {::LinearMap::SCALAR, &Scalar},
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


}  // namespace linear_map
