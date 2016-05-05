#ifndef EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H
#define EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "epsilon/linear/linear_map.h"
#include "epsilon/vector/vector_util.h"

namespace linear_map {

class SparseMatrixImpl final : public LinearMapImpl {
 public:
  SparseMatrixImpl(SparseMatrix A)
      : LinearMapImpl(SPARSE_MATRIX), A_(A) {
    CHECK(A_.isCompressed());
  }

  int m() const override { return A_.rows(); }
  int n() const override { return A_.cols(); }
  std::string DebugString() const override;
  DenseMatrix AsDense() const override { return static_cast<DenseMatrix>(A_); }

  DenseVector Apply(const DenseVector& x) const override { return A_*x; }

  LinearMapImpl* Transpose() const override {
    return new SparseMatrixImpl(A_.transpose());
  }
  LinearMapImpl* Inverse() const override;

  bool operator==(const LinearMapImpl& other) const override;

  // Sparse matrix API
  const SparseMatrix& sparse() const { return A_; }

 private:
  SparseMatrix A_;
};

}  // namespace linear_map

#endif  // EPSILON_LINEAR_SPARSE_MATRIX_IMPL_H
