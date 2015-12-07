#ifndef EPSILON_LINEAR_SCALAR_MATRIX_IMPL_H
#define EPSILON_LINEAR_SCALAR_MATRIX_IMPL_H

#include "epsilon/linear/linear_map.h"
#include "epsilon/vector/vector_util.h"
#include "epsilon/util/string.h"

namespace linear_map {

class ScalarMatrixImpl final : public LinearMapImpl {
 public:
  ScalarMatrixImpl(int n, double alpha)
      : LinearMapImpl(SCALAR_MATRIX), n_(n), alpha_(alpha) {}

  int m() const override { return n_; }
  int n() const override { return n_; }
  std::string DebugString() const override {
    return StringPrintf(
        "scalar matrix: n=%d alpha=%3.4f", n_, alpha_);
  }
  DenseMatrix AsDense() const override {
    return alpha_*DenseMatrix::Identity(n_, n_);
  }
  DenseMatrix ApplyMatrix(const DenseMatrix& X) const override { return alpha_*X; }
  DenseVector Apply(const DenseVector& x) const override { return alpha_*x; }

  LinearMapImpl* Transpose() const override {
    return new ScalarMatrixImpl(n_, alpha_);
  }
  LinearMapImpl* Inverse() const override {
    return new ScalarMatrixImpl(n_, 1/alpha_);
  }

  bool operator==(const LinearMapImpl& other) const override;

  // Scalar matrix API
  double alpha() const { return alpha_; }
  SparseMatrix AsSparse() const;

 private:
  int n_;
  double alpha_;
};

}  // namespace linear_map

#endif  // EPSILON_LINEAR_SCALAR_MATRIX_IMPL_H
