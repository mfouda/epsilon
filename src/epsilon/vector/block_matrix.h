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
#include "epsilon/vector/matrix_variant.h"

class BlockMatrix {
 public:
  class Solver {
   public:
    BlockVector solve(const BlockVector& b);
  };

  MatrixVariant& operator()(
      const std::string& row_key, const std::string& col_key);
  BlockMatrix& operator*=(const BlockMatrix& rhs);

  friend BlockVector operator*(const BlockMatrix& A, const BlockVector& x);

  BlockMatrix transpose() const;
  Solver inv(double lambda = 0) const;

 private:

  // col -> row -> value
  std::map<std::string, std::map<std::string, MatrixVariant>> data_;
};

BlockMatrix operator*(BlockMatrix lhs, const BlockMatrix& rhs);
BlockVector operator*(const BlockMatrix& lhs, const BlockVector& rhs);

#endif  // EPSILON_VECTOR_BLOCK_MATRIX_H
