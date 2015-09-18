#include "epsilon/vector/vector_util.h"

#include <fstream>

#include <glog/logging.h>

#include "epsilon/util/string.h"

using Eigen::Map;
using Eigen::VectorXd;

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
  Eigen::SparseMatrix<double, Eigen::RowMajor> I(n, n);
  I.setIdentity();
  for (SparseXd::InnerIterator iter(A, 0); iter; ++iter) {
    CHECK(iter.value() != 0);
    if (iter.row() + n > m)
      return false;
    B.middleRows(iter.row(), n) -= iter.value()*I;
  }
  B.prune([](int, int, float val) { return val != 0; });
  return B.nonZeros() == 0;
  return false;
}
