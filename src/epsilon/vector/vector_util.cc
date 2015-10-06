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

MatrixXd VStack(const MatrixXd& A, const MatrixXd& B) {
  MatrixXd X(A.rows() + B.rows(), B.cols());
  if (A.rows() > 0) {
    CHECK_EQ(A.cols(), B.cols());
    X.topRows(A.rows()) = A;
  }
  X.bottomRows(B.rows()) = B;
  return X;
}

MatrixXd HStack(const MatrixXd& A, const MatrixXd& B) {
  MatrixXd X(A.rows(), A.cols() + B.cols());
  if (A.cols() > 0) {
    CHECK_EQ(A.rows(), B.rows());
    X.leftCols(A.cols()) = A;
  }
  X.rightCols(B.cols()) = B;
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

bool IsScalarMatrix(const SparseXd& A, double* alpha) {
  if (!IsDiagonal(A))
    return false;
  if ((A.diagonal().array() == A.coeff(0,0)).all()) {
    *alpha = A.coeff(0,0);
    return true;
  }
  return false;
}

Eigen::VectorXd ToVector(const Eigen::MatrixXd& A) {
  return Eigen::Map<const Eigen::VectorXd>(A.data(), A.rows()*A.cols());
}

Eigen::MatrixXd ToMatrix(const Eigen::VectorXd& a, int m, int n) {
  CHECK_EQ(a.size(), m*n);
  return Eigen::Map<const Eigen::MatrixXd>(a.data(), m, n);
}

bool IsMatrixEqual(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
  return (A.rows() == B.rows() &&
          A.cols() == B.cols() &&
          A == B);
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
