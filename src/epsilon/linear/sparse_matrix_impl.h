#ifndef EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H
#define EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H

#include <Eigen/Sparse>

#include "epsilon/linear/linear_map.h"
#include "epsilon/vector/vector_util.h"

class SparseMatrixImpl final : public LinearMapImpl {
 public:
  SparseMatrixImpl(SparseMatrix A)
      : LinearMapImpl(SPARSE_MATRIX), A_(A) {}

  virtual int m() const { return A_.rows(); }
  virtual int n() const { return A_.cols(); }
  virtual std::string DebugString() const { return SparseMatrixDebugString(A_); }
  virtual DenseMatrix AsDense() const { return static_cast<DenseMatrix>(A_); }
  virtual DenseVector Apply(const DenseVector& x) const { return A_*x; }

  virtual std::unique_ptr<LinearMapImpl> Transpose() const {
    return std::unique_ptr<LinearMapImpl>(new SparseMatrixImpl(A_.transpose()));
  }
  virtual std::unique_ptr<LinearMapImpl> Inverse() const {
    LOG(FATAL) << "Not implemented";
  }

  // Sparse matrix API
  const SparseMatrix sparse() const { return A_; }

 private:
  SparseMatrix A_;
};

#endif  // EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H
