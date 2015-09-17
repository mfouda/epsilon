#include "epsilon/vector/vector_file.h"

#include <glog/logging.h>

#include "epsilon/file/file.h"

const std::string kMetadataFile = "metadata";
const std::string kValueFile = "value";
const std::string kValueTransposeFile = "value_transpose";

std::string metadata_file(const std::string& location) {
  return location + "/" + kMetadataFile;
}

std::string value_file(const std::string& location) {
  return location + "/" + kValueFile;
}

std::string value_transpose_file(const std::string& location) {
  return location + "/" + kValueTransposeFile;
}

void WriteMatrixData(const Eigen::MatrixXd& input, const std::string& location) {
  const int m = input.rows();
  const int n = input.cols();

  {
    // Write metadata
    Data data;
    data.set_data_type(Data::DENSE_MATRIX);
    data.mutable_dense_matrix()->set_m(m);
    data.mutable_dense_matrix()->set_n(n);
    std::unique_ptr<file::File> file = file::Open(metadata_file(location), "w");
    file->Write(data.SerializeAsString());
    file->Close();
  }

  {
    // Write value
    std::unique_ptr<file::File> file = file::Open(value_file(location), "w");
    const std::string value_str(
        reinterpret_cast<const char*>(input.data()), sizeof(double)*m*n);
    file->Write(value_str);
    file->Close();
  }

  {
    // Write value transpose
    std::unique_ptr<file::File> file = file::Open(value_transpose_file(location), "w");
    const std::string value_str(
        reinterpret_cast<const char*>(
            static_cast<Eigen::MatrixXd>(input.transpose()).data()),
        sizeof(double)*m*n);
    file->Write(value_str);
    file->Close();
  }
}

std::unique_ptr<const Data> ReadSplitData(const std::string& location) {
  Data* data = new Data;
  {
    // Read metadata
    std::unique_ptr<file::File> file = file::Open(metadata_file(location));
    CHECK(data->ParseFromString(file->Read()));
    file->Close();
  }

  {
    // Read value
    std::unique_ptr<file::File> file = file::Open(value_file(location));
    data->set_value(file->Read());
    file->Close();
  }
  return std::unique_ptr<const Data>(data);
}

Eigen::MatrixXd GetMatrixData(const Data& data) {
  CHECK_EQ(data.data_type(), Data::DENSE_MATRIX);
  const int m = data.dense_matrix().m();
  const int n = data.dense_matrix().n();

  CHECK_EQ(m*n*sizeof(double), data.value().size());
  return Eigen::Map<const Eigen::MatrixXd>(
    reinterpret_cast<const double*>(data.value().data()), m, n);
}
