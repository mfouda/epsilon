#ifndef EPSILON_LINEAR_DENSE_MATRIX_IMPL_H
#define EPSILON_LINEAR_DENSE_MATRIX_IMPL_H

#include <glog/logging.h>

#include "epsilon/linear/linear_map.h"

namespace linear_map {

class DenseMatrixImpl final : public LinearMapImpl {
 public:
  DenseMatrixImpl(DenseMatrix A) : LinearMapImpl(DENSE_MATRIX), A_(A) {}

  int m() const override { return A_.rows(); }
  int n() const override { return A_.cols(); }
  std::string DebugString() const override;
  DenseMatrix AsDense() const override { return A_; }

  DenseVector Apply(const DenseVector& x) const override { return A_*x; }

  LinearMapImpl* Transpose() const override;
  LinearMapImpl* Inverse() const override;

  bool operator==(const LinearMapImpl& other) const override;

  // Dense matrix API
  const DenseMatrix& dense() const { return A_; }

 private:
  DenseMatrix A_;
};

}  // namespace linear_map

#endif  // EPSILON_LINEAR_DENSE_MATRIX_IMPL_H
