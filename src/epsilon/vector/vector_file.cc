#include "epsilon/vector/vector_file.h"

#include <glog/logging.h>

#include "epsilon/file/file.h"

Eigen::MatrixXd ReadMatrixData(const Constant& constant) {
  CHECK_EQ(constant.constant_type(), Constant::DENSE_MATRIX);
  const int m = constant.m();
  const int n = constant.n();

  std::unique_ptr<file::File> file = file::Open(constant.data_location(), "r");
  std::string data_str = file->Read();
  CHECK_EQ(m*n*sizeof(double), data_str.size());
  return Eigen::Map<const Eigen::MatrixXd>(
      reinterpret_cast<const double*>(data_str.data()), m, n);
}
