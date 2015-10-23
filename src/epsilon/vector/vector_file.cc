#include "epsilon/vector/vector_file.h"

#include <glog/logging.h>

#include "epsilon/file/file.h"

// Eigen::MatrixXd GetMatrixData(const Data& data) {
//   CHECK_EQ(data.data_type(), Data::DENSE_MATRIX);
//   const int m = data.m();
//   const int n = data.n();

//   CHECK_EQ(m*n*sizeof(double), data.value().size());
//   return Eigen::Map<const Eigen::MatrixXd>(
//     reinterpret_cast<const double*>(data.value().data()), m, n);
// }
