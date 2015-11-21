// Block matrix where each element is a matrix variant.
//
// Usage:
//
// BlockVector x;
// x("col1") = Eigen::VectorXd::Random(10)
//
// BlockVector y;
// y("row1") = Eigen::VectorXd::Zero(4);
// y("row3") = Eigen::VectorXd::Ones(3);
//
// BlockMatrixVariant A;
// A("row1", "col1") += Eigen::MatrixXd::Identity(10);
// A("row3", "col1") += B
// ...
//
// Matrix-vector products:
// y += A*x;
// x += A.transpose()*y;
//
// Matrix-matrix products and inverse
// ATA = A.transpose()*A
// ATAinv = ATA.solver(lam)

#ifndef EPSILON_VECTOR_BLOCK_MATRIX_H
#define EPSILON_VECTOR_BLOCK_MATRIX_H

#include <set>

#include "epsilon/vector/block_vector.h"
#include "epsilon/linear/linear_map.h"

class BlockMatrix {
 public:
  linear_map::LinearMap& operator()(
      const std::string& row_key, const std::string& col_key);
  const linear_map::LinearMap& operator()(
      const std::string& row_key, const std::string& col_key) const;

  std::string DebugString() const;
  int m() const;
  int n() const;
  const std::map<std::string, linear_map::LinearMap>& col(
      const std::string& col_key) const;
  const std::map<std::string, std::map<std::string, linear_map::LinearMap>>&
      data() const { return data_; }

  std::set<std::string> col_keys() const;
  std::set<std::string> row_keys() const;

  BlockMatrix Transpose() const;
  BlockMatrix Inverse() const;

  // Returns a matrix such that IA = A or AI = A.
  BlockMatrix LeftIdentity() const;
  BlockMatrix RightIdentity() const;

  void InsertOrAdd(const std::string& row_key, const std::string& col_key,
                   linear_map::LinearMap value);

 private:
  // col -> row -> valu
  std::map<std::string, std::map<std::string, linear_map::LinearMap>> data_;

  friend BlockMatrix operator*(const BlockMatrix& A, const BlockMatrix& B);
  friend BlockMatrix operator*(double alpha, const BlockMatrix& A);
  friend BlockMatrix operator+(const BlockMatrix& A, const BlockMatrix& B);

  friend BlockVector operator*(const BlockMatrix& A, const BlockVector& x);
};

// Matrix-matrix product, sum
BlockMatrix operator*(const BlockMatrix& lhs, const BlockMatrix& rhs);
BlockMatrix operator+(const BlockMatrix& lhs, const BlockMatrix& rhs);
BlockMatrix operator-(const BlockMatrix& lhs, const BlockMatrix& rhs);

// Matrix-matrix short hand products
BlockMatrix operator*(double alpha, const BlockMatrix& A);
BlockMatrix operator*(const BlockMatrix& A, double alpha);

// Matrix-vector product
BlockVector operator*(const BlockMatrix& lhs, const BlockVector& rhs);

#endif  // EPSILON_VECTOR_BLOCK_MATRIX_H
