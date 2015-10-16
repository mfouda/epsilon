#ifndef EPSILON_LINEAR_DENSE_MATRIX_IMPL_H
#define EPSILON_LINEAR_DENSE_MATRIX_IMPL_H

#include <glog/logging.h>

#include "epsilon/linear/linear_map.h"
#include "epsilon/vector/vector_util.h"

class DenseMatrixImpl final : public LinearMapImpl {
 public:
  DenseMatrixImpl(DenseMatrix A) : LinearMapImpl(DENSE_MATRIX), A_(A) {}

  virtual int m() const { return A_.rows(); }
  virtual int n() const { return A_.cols(); }
  virtual std::string DebugString() const { return MatrixDebugString(A_); }
  virtual DenseMatrix AsDense() const { return A_; }
  virtual DenseVector Apply(const DenseVector& x) const { return A_*x; }

  virtual std::unique_ptr<LinearMapImpl> Transpose() const {
    return std::unique_ptr<LinearMapImpl>(new DenseMatrixImpl(A_.transpose()));
  }
  virtual std::unique_ptr<LinearMapImpl> Inverse() const {
    LOG(FATAL) << "Not implemented";
  }

  // Dense matrix API
  const DenseMatrix& dense() const { return A_; }

 private:
  DenseMatrix A_;
};

#endif  // EPSILON_LINEAR_DENSE_MATRIX_IMPL_H
