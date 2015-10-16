#ifndef EPSILON_LINEAR_DIAGONAL_MATRIX_IMPL_H
#define EPSILON_LINEAR_DIAGONAL_MATRIX_IMPL_H

#include <Eigen/Dense>

#include "epsilon/linear/linear_map.h"
#include "epsilon/vector/vector_util.h"

class DiagonalMatrixImpl final : public LinearMapImpl {
 public:
  typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagonalMatrix;

  DiagonalMatrixImpl(DiagonalMatrix A)
      : LinearMapImpl(DIAGONAL_MATRIX), A_(A) {}

  int m() const override { return A_.rows(); }
  int n() const override { return A_.cols(); }
  std::string DebugString() const override { return MatrixDebugString(A_); }
  DenseMatrix AsDense() const override { return static_cast<DenseMatrix>(A_); }
  DenseVector Apply(const DenseVector& x) const override { return A_*x; }

  std::unique_ptr<LinearMapImpl> Transpose() const override {
    return std::unique_ptr<LinearMapImpl>(new DiagonalMatrixImpl(A_));
  }
  std::unique_ptr<LinearMapImpl> Inverse() const override {
    LOG(FATAL) << "Not implemented";
  }

  // Diagonal matrix API
  const DiagonalMatrix& diagonal() const { return A_; }

 private:
  DiagonalMatrix A_;
};

#endif  // EPSILON_LINEAR_DIAGONAL_MATRIX_IMPL_H
