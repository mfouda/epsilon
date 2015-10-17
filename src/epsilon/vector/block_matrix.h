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

#include "epsilon/vector/block_vector.h"
#include "epsilon/linear/linear_map.h"

class BlockMatrix {
 public:
  BlockMatrix() {
    VLOG(3) << "default ctor";
  }

  ~BlockMatrix() {
    VLOG(3) << "dtor";
  }

  LinearMap& operator()(
      const std::string& row_key, const std::string& col_key);
  const LinearMap& operator()(
      const std::string& row_key, const std::string& col_key) const;

  friend BlockMatrix operator*(const BlockMatrix& A, const BlockMatrix& B);
  friend BlockVector operator*(const BlockMatrix& A, const BlockVector& x);

  std::string DebugString() const;
  int m() const;
  int n() const;
  const std::map<std::string, LinearMap>& col(
      const std::string& col_key) const;
  std::vector<std::string> col_keys() const;

  BlockMatrix Transpose() const;
  BlockMatrix Inverse() const;

  void InsertOrAdd(const std::string& row_key, const std::string& col_key,
                   LinearMap value);

 private:

  // col -> row -> value
  std::map<std::string, std::map<std::string, LinearMap>> data_;
};

BlockMatrix operator*(const BlockMatrix& lhs, const BlockMatrix& rhs);
BlockVector operator*(const BlockMatrix& lhs, const BlockVector& rhs);

#endif  // EPSILON_VECTOR_BLOCK_MATRIX_H
