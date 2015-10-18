#ifndef EPSILON_LINEAR_DENSE_MATRIX_IMPL_H
#define EPSILON_LINEAR_DENSE_MATRIX_IMPL_H

#include <glog/logging.h>

#include "epsilon/linear/linear_map.h"
#include "epsilon/vector/vector_util.h"

class DenseMatrixImpl final : public LinearMapImpl {
 public:
  DenseMatrixImpl(DenseMatrix A) : LinearMapImpl(DENSE_MATRIX), A_(A) {}

  int m() const override { return A_.rows(); }
  int n() const override { return A_.cols(); }
  std::string DebugString() const override { return MatrixDebugString(A_); }
  DenseMatrix AsDense() const override { return A_; }
  DenseVector Apply(const DenseVector& x) const override { return A_*x; }

  LinearMapImpl* Transpose() const override {
    return new DenseMatrixImpl(A_.transpose());
  }
  LinearMapImpl* Inverse() const override;

  // Dense matrix API
  const DenseMatrix& dense() const { return A_; }

 private:
  DenseMatrix A_;
};

#endif  // EPSILON_LINEAR_DENSE_MATRIX_IMPL_H
