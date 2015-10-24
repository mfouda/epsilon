#include "epsilon/vector/vector_file.h"

#include <glog/logging.h>

#include "epsilon/file/file.h"
#include "epsilon/vector/vector_util.h"

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

Eigen::SparseMatrix<double> ReadSparseMatrixData(const Constant& constant) {
  CHECK_EQ(constant.constant_type() , Constant::SPARSE_MATRIX);
  std::unique_ptr<file::File> file = file::Open(constant.data_location(), "r");
  std::string data_str = file->Read();

  const int m = constant.m();
  const int n = constant.n();
  const int nnz = constant.nnz();

  CHECK_EQ(nnz*sizeof(double) + (n+nnz+1)*sizeof(int32_t), data_str.size());
  const int32_t* col_ptr = reinterpret_cast<const int32_t*>(data_str.data());
  const int32_t* row_index = col_ptr + n+1;
  const double* values = reinterpret_cast<const double*>(row_index + nnz);

  return Eigen::MappedSparseMatrix<double>(m, n, nnz,
      const_cast<int32_t*>(col_ptr),
      const_cast<int32_t*>(row_index),
      const_cast<double*>(values));
}
