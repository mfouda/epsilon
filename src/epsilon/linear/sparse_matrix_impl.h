#ifndef EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H
#define EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H

#include <Eigen/Sparse>

#include "epsilon/linear/linear_map.h"
#include "epsilon/vector/vector_util.h"

class SparseMatrixImpl final : public LinearMapImpl {
 public:
  SparseMatrixImpl(SparseMatrix A)
      : LinearMapImpl(SPARSE_MATRIX), A_(A) {}

  int m() const override { return A_.rows(); }
  int n() const override { return A_.cols(); }
  std::string DebugString() const override { return SparseMatrixDebugString(A_); }
  DenseMatrix AsDense() const override { return static_cast<DenseMatrix>(A_); }
  DenseVector Apply(const DenseVector& x) const override { return A_*x; }

  std::unique_ptr<LinearMapImpl> Transpose() const override {
    return std::unique_ptr<LinearMapImpl>(new SparseMatrixImpl(A_.transpose()));
  }
  std::unique_ptr<LinearMapImpl> Inverse() const override {
    LOG(FATAL) << "Not implemented";
  }

  // Sparse matrix API
  const SparseMatrix& sparse() const { return A_; }

 private:
  SparseMatrix A_;
};

#endif  // EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H
