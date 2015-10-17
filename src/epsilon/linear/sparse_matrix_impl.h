#ifndef EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H
#define EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

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

  LinearMapImpl* Transpose() const override {
    return new SparseMatrixImpl(A_.transpose());
  }
  LinearMapImpl* Inverse() const override {
    // TODO(mwytock): Verify symmetry, fill-in
    CHECK_EQ(A_.rows(), A_.cols());

    Eigen::SimplicialLDLT<SparseMatrix> ldlt_;
    ldlt_.compute(A_);
    CHECK_EQ(Eigen::Success, ldlt_.info());
    SparseMatrix A_inv = ldlt_.solve(SparseIdentity(A_.rows()));
    return new SparseMatrixImpl(A_inv);
  }

  // Sparse matrix API
  const SparseMatrix& sparse() const { return A_; }

 private:
  SparseMatrix A_;
};

#endif  // EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H
