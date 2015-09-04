#include "distopt/util/vector.h"

#include <fstream>

#include <glog/logging.h>

#include "distopt/util/string.h"
#include "distopt/file/file.h"

using Eigen::Map;
using Eigen::VectorXd;
using google::protobuf::RepeatedFieldBackInserter;

VectorXd GetVector(const Vector& input, int start, int size) {
  if (size == -1) size = input.value().size() - start;
  return Map<const VectorXd>(input.value().data() + start, size);
}

SparseXd GetSparseMatrix(const SparseMatrix& input) {
  std::vector<Eigen::Triplet<double> > coeffs;
  for (int j = 0; j < input.n(); j++) {
    for (int k = input.jc(j); k < input.jc(j+1); k++) {
      coeffs.push_back(Eigen::Triplet<double>(input.ir(k), j, input.pr(k)));
    }
  }

  SparseXd output(input.m(), input.n());
  output.setFromTriplets(coeffs.begin(), coeffs.end());
  return output;
}

void GetVectorProto(const Eigen::VectorXd& input, Vector* output) {
  std::copy_n(input.data(), input.rows(),
              RepeatedFieldBackInserter(output->mutable_value()));
}

void GetMatrixProto(const Eigen::MatrixXd& input, DenseMatrix* output) {
  std::copy_n(input.data(), input.cols()*input.rows(),
              RepeatedFieldBackInserter(output->mutable_value()));
  output->set_m(input.rows());
  output->set_n(input.cols());
}

SparseXd BlockDiag(const MatrixXd& A, int count) {
  std::vector<Eigen::Triplet<double> > coeffs;
  for (int k = 0; k < count; k++) {
    for (int j = 0; j < A.cols(); j++) {
      for (int i = 0; i < A.rows(); i++) {
        coeffs.push_back(Eigen::Triplet<double>(k*A.rows()+i, k*A.cols()+j, A(i,j)));
      }
    }
  }

  SparseXd B(A.rows()*count, A.cols()*count);
  B.setFromTriplets(coeffs.begin(), coeffs.end());
  return B;
}

SparseXd RandomSparse(int m, int n, double d) {
  int nnz = ceil(m*n*d);
  std::vector<Eigen::Triplet<double> > coeffs;
  for (int k = 0; k < nnz; k++) {
    coeffs.push_back(Eigen::Triplet<double>(
        rand() % m, rand() % n, static_cast<double>(rand()) / RAND_MAX));
  }

  SparseXd A(m, n);
  A.reserve(coeffs.size());
  A.setFromTriplets(coeffs.begin(), coeffs.end());

  return A;
}

SparseXd DiagonalSparse(const VectorXd& a) {
  const int n = a.size();
  std::vector<Eigen::Triplet<double> > coeffs(n);
  for (int i = 0; i < n; i++)
    coeffs[i] = Eigen::Triplet<double>(i, i, a(i));
  SparseXd A(n, n);
  A.reserve(n);
  A.setFromTriplets(coeffs.begin(), coeffs.end());
  return A;
}

SparseXd SparseIdentity(int n) {
  SparseXd A(n, n);
  A.setIdentity();
  return A;
}

MatrixXd GetMatrix(const DenseMatrix& input) {
  CHECK_EQ(input.m()*input.n(), input.value().size());
  return Map<const MatrixXd>(input.value().data(), input.m(), input.n());
}

MatrixXd GetMatrixData(const Data& data) {
  CHECK_EQ(data.data_type(), Data::DENSE_MATRIX);
  const int m = data.dense_matrix().m();
  const int n = data.dense_matrix().n();

  CHECK_EQ(m*n*sizeof(double), data.value().size());
  return Eigen::Map<const MatrixXd>(
    reinterpret_cast<const double*>(data.value().data()), m, n);
}

void AppendBlockTriplets(
    const MatrixXd& A, int i_offset, int j_offset,
    std::vector<Eigen::Triplet<double> >* coeffs) {
  for (int j = 0; j < A.cols(); j++) {
    for (int i = 0; i < A.rows(); i++) {
      if (A(i,j) != 0)
        coeffs->push_back(Eigen::Triplet<double>(i_offset+i, j_offset+j, A(i,j)));
    }
  }
}

void AppendBlockTriplets(
    const SparseXd& A, int i_offset, int j_offset,
    std::vector<Eigen::Triplet<double> >* coeffs) {
  for (int k = 0; k < A.outerSize(); k++) {
    for (SparseXd::InnerIterator iter(A, k); iter; ++iter) {
      coeffs->push_back(
          Eigen::Triplet<double>(i_offset+iter.row(), j_offset+iter.col(),
                                 iter.value()));
    }
  }
}

void GetSparseMatrixProto(const SparseXd& input, SparseMatrix* output) {
  // TODO(mwytock): Check that its column major ordering
  output->set_m(input.rows());
  output->set_n(input.cols());
  std::copy_n(input.valuePtr(), input.nonZeros(),
              RepeatedFieldBackInserter(output->mutable_pr()));
  std::copy_n(input.outerIndexPtr(), input.cols()+1,
              RepeatedFieldBackInserter(output->mutable_jc()));
  std::copy_n(input.innerIndexPtr(), input.nonZeros(),
              RepeatedFieldBackInserter(output->mutable_ir()));
}

Data GetSparseMatrixDataProto(const SparseXd& input) {
  Data data;
  data.set_data_type(Data::SPARSE_MATRIX);
  GetSparseMatrixProto(input, data.mutable_sparse_matrix());
  return data;
}

VectorXd RowNorm(const SparseXd& A) {
  return static_cast<VectorXd>(A.cwiseProduct(A)*VectorXd::Ones(A.cols()))
      .array().sqrt();
}

VectorXd ColNorm(const SparseXd& A) {
  return static_cast<VectorXd>(A.cwiseProduct(A).transpose()*
                               VectorXd::Ones(A.rows()))
      .array().sqrt();
}

MatrixXd Stack(const MatrixXd& A, const MatrixXd& B) {
  MatrixXd X(A.rows() + B.rows(), B.cols());
  if (A.rows() > 0) {
    CHECK_EQ(A.cols(), B.cols());
    X.topRows(A.rows()) = A;
  }
  X.bottomRows(B.rows()) = B;
  return X;
}

void WriteTextSparseMatrix(const SparseXd& input, const std::string& file) {
  std::ofstream out(file, std::ios::out);
  for (int i = 0; i < input.rows(); i++) {
    for (int j = 0; j < input.cols(); j++) {
      out << input.coeff(i,j);
      out << " ";
    }
    out << "\n";
  }
  out.close();
}

void WriteTextVector(const VectorXd& input, const std::string& file) {
  std::ofstream out(file, std::ios::out);
  for (int i = 0; i < input.size(); i++) {
    out << input(i);
    out << "\n";
  }
  out.close();
}

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

void WriteMatrixData(const MatrixXd& input, const std::string& location) {
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
            static_cast<MatrixXd>(input.transpose()).data()),
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

MatrixXd ReadMatrixRows(const std::string& location, int start, int num_rows) {
  Data data;
  {
    // Read metadata
    std::unique_ptr<file::File> file = file::Open(metadata_file(location));
    CHECK(data.ParseFromString(file->Read()));
    file->Close();
  }

  CHECK_EQ(data.data_type(), Data::DENSE_MATRIX);
  const int m = data.dense_matrix().m();
  const int n = data.dense_matrix().n();
  CHECK_LT(start, m);
  CHECK_LE(start + num_rows, m);

  // Read value in row order and construct matrix
  std::unique_ptr<file::File> file = file::Open(
      value_transpose_file(location));
  return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
      reinterpret_cast<const double*>(
          file->Read(start*n*sizeof(double),
                     num_rows*n*sizeof(double)).data()),
      num_rows, n);
}

std::string VectorDebugString(const VectorXd& x) {
  std::string retval("[");

  const int max_elems = 10;
  if (x.size() > max_elems) {
    for (int i = 0; i < max_elems/2; i++) {
      retval += StringPrintf("%.4f ", x[i]);
    }

    retval += "... ";

    for (int i = x.size()-max_elems/2-1; i < x.size(); i++) {
      retval += StringPrintf("%.4f ", x[i]);
    }
  } else {
    for (int i = 0; i < x.size(); i++) {
      retval += StringPrintf("%.4f ", x[i]);
    }
  }

  if (retval.size() > 1) {
    retval = retval.substr(0, retval.size()-1) + "]";
  } else {
    retval += "]";
  }

  return retval;
}

std::string MatrixDebugString(const MatrixXd& A) {
  std::string retval;

  const int max_rows = 10;
  if (A.rows() > max_rows) {
    for (int i = 0; i < max_rows/2; i++) {
      retval += VectorDebugString(A.row(i)) + "\n";
    }
    retval += "...\n";
    for (int i = A.rows()-max_rows/2-1; i < A.rows(); i++) {
      retval += VectorDebugString(A.row(i)) + "\n";
    }
  } else {
    for (int i = 0; i < A.rows(); i++) {
      retval += VectorDebugString(A.row(i)) + "\n";
    }
  }

  return retval;
}

std::string SparseMatrixDebugString(const SparseXd& A) {
  std::string retval;

  const int max_rows = 10;
  if (A.rows() > max_rows) {
    for (int i = 0; i < max_rows/2; i++) {
      retval += VectorDebugString(A.row(i)) + "\n";
    }
    retval += "...\n";
    for (int i = A.rows()-max_rows/2-1; i < A.rows(); i++) {
      retval += VectorDebugString(A.row(i)) + "\n";
    }
  } else {
    for (int i = 0; i < A.rows(); i++) {
      retval += VectorDebugString(A.row(i)) + "\n";
    }
  }

  return retval;
}

bool IsDiagonal(const SparseXd& A) {
  for (int k = 0; k < A.outerSize(); k++) {
    for (SparseXd::InnerIterator iter(A, k); iter; ++iter) {
      if (iter.row() != iter.col())
        return false;
    }
  }
  return true;
}

Eigen::VectorXd ToVector(const Eigen::MatrixXd& A) {
  return Eigen::Map<const Eigen::VectorXd>(A.data(), A.rows()*A.cols());
}

Eigen::MatrixXd ToMatrix(const Eigen::VectorXd& a, int m, int n) {
  CHECK_EQ(a.size(), m*n);
  return Eigen::Map<const Eigen::MatrixXd>(a.data(), m, n);
}

bool IsBlockScalar(const SparseXd& A) {
  CHECK(!A.IsRowMajor);

  const int m = A.rows();
  const int n = A.cols();
  if (m < n)
    return false;

  // Go through first column, subtract off (potential) identity
  // matrices and then check if resulting matrix is all zeros
  Eigen::SparseMatrix<double, Eigen::RowMajor> B = A;
  for (SparseXd::InnerIterator iter(A, 0); iter; ++iter) {
    CHECK(iter.value() != 0);
    if (iter.row() + n > m)
      return false;
    B.middleRows(iter.row(), n) -= iter.value()*SparseIdentity(n);
  }
  B.prune([](int, int, float val) { return val != 0; });
  return B.nonZeros() == 0;
}
