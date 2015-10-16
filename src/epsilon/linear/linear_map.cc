
#include "epsilon/linear/scalar_matrix_impl.h"

LinearMap::LinearMap() : impl_(new ScalarMatrixImpl(0, 0)) {}
