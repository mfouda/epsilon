
#include "epsilon/linear/scalar_matrix_impl.h"

LinearMap::LinearMap() : impl_(new ScalarMatrixImpl(0, 0)) {}

LinearMap LinearMap::Identity(int n) {
  return LinearMap(new ScalarMatrixImpl(n, 1));
}

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
