
#include "epsilon/linear/scalar_matrix_impl.h"

LinearMap::LinearMap() : impl_(new ScalarMatrixImpl(0, 0)) {}

LinearMap LinearMap::Identity(int n) {
  return LinearMap(new ScalarMatrixImpl(n, 1));
}
