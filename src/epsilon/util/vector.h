#ifndef UTIL_VECTOR_H
#define UTIL_VECTOR_H

#include <memory>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "distopt/data.pb.h"

using Eigen::CwiseNullaryOp;
using Eigen::EigenBase;
using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef Eigen::SparseMatrix<double> SparseXd;
typedef void(*VectorFunction)(
    const std::vector<const VectorXd*>& input,
    const std::vector<VectorXd*>& output);

// Eigen objects from protos
VectorXd GetVector(const Vector& input, int start = 0, int size = -1);
MatrixXd GetMatrix(const DenseMatrix& input);
SparseXd GetSparseMatrix(const SparseMatrix& input);

// Eigen objects to protos
void GetVectorProto(const VectorXd& input, Vector* output);
void GetMatrixProto(const MatrixXd& input, DenseMatrix* output);
void GetSparseMatrixProto(const SparseXd& input, SparseMatrix* output);

// Create a block diagonal matrix with A repeated k times
SparseXd BlockDiag(const MatrixXd& A, int k);
SparseXd RandomSparse(int m, int n, double d);
SparseXd DiagonalSparse(const VectorXd& a);
SparseXd SparseIdentity(int n);

bool IsDiagonal(const SparseXd& A);

// True if A = [ aI; 0; bI; ...]
bool IsBlockScalar(const SparseXd& A);

// Extract the coefficients for an sparse or dense matrix
void AppendBlockTriplets(const SparseXd& A, int i, int j,
                         std::vector<Eigen::Triplet<double> >* coeffs);
void AppendBlockTriplets(const MatrixXd& A, int i, int j,
                         std::vector<Eigen::Triplet<double> >* coeffs);

// Row and column norms of sparse matrices
VectorXd RowNorm(const SparseXd& A);
VectorXd ColNorm(const SparseXd& A);

// X = [A; B]
MatrixXd Stack(const MatrixXd& A, const MatrixXd& B);

// Debugging stuff

// Write matrices in text format for debugging
void WriteTextMatrix(const SparseXd& input, const std::string& file);
void WriteTextSparseMatrix(const SparseXd& input, const std::string& file);
void WriteTextVector(const VectorXd& input, const std::string& file);

std::string VectorDebugString(const VectorXd& x);
std::string MatrixDebugString(const MatrixXd& A);
std::string SparseMatrixDebugString(const SparseXd& A);

// Read/write matrices/vectors in binary format with split metadata + value
// format

// Write path
void WriteMatrixData(const MatrixXd& input, const std::string& location);

// Read path is broken up in to two steps currently
std::unique_ptr<const Data> ReadSplitData(const std::string& location);
MatrixXd GetMatrixData(const Data& data);
MatrixXd ReadMatrixRows(const std::string& location, int start_row, int n);

std::string metadata_file(const std::string& location);
std::string value_file(const std::string& location);
std::string value_transpose_file(const std::string& location);

// vec()/mat() operators
Eigen::VectorXd ToVector(const Eigen::MatrixXd& A);
Eigen::MatrixXd ToMatrix(const Eigen::VectorXd& a, int m, int n);

#endif  // UTIL_VECTOR_H
