#ifndef EPSILON_LINEAR_KRONECKER_PRODUCT_IMPL_H
#define EPSILON_LINEAR_KRONECKER_PRODUCT_IMPL_H

#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/util/string.h"
#include "epsilon/vector/vector_util.h"

class KroneckerProductImpl final : public LinearMapImpl {
 public:
  KroneckerProductImpl(
      LinearMapImpl* A,
      LinearMapImpl* B)
      : LinearMapImpl(KRONECKER_PRODUCT), A_(A), B_(B) {}

  int m() const override { return A_.impl().m()*B_.impl().m(); }
  int n() const override { return A_.impl().n()*B_.impl().n(); }

  std::string DebugString() const override {
    return StringPrintf("kronecker product\nA: %s\nB: %s",
                        A_.impl().DebugString().c_str(),
                        B_.impl().DebugString().c_str());
  }

  DenseMatrix AsDense() const override {
    DenseMatrix A = A_.impl().AsDense();
    DenseMatrix B = B_.impl().AsDense();
    DenseMatrix C(m(), n());

    for (int i = 0; i < A.rows(); i++) {
      for (int j = 0; j < A.cols(); j++) {
        C.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j)*B;
      }
    }

    return C;
  }

  DenseVector Apply(const DenseVector& x) const override {
    LinearMap X(new DenseMatrixImpl(ToMatrix(x, B_.impl().n(), A_.impl().n())));
    LinearMap Y = (A_*(B_*X).Transpose()).Transpose();
    return ToVector(Y.impl().AsDense());
  }

  LinearMapImpl* Transpose() const override {
    return new KroneckerProductImpl(
        A_.impl().Transpose(),
        B_.impl().Transpose());
  }

  LinearMapImpl* Inverse() const override {
    return new KroneckerProductImpl(
        A_.impl().Inverse(),
        B_.impl().Inverse());
  }

  // Scalar matrix API
  const LinearMap& A() const { return A_; }
  const LinearMap& B() const { return B_; }

 private:
  LinearMap A_;
  LinearMap B_;
};

#endif  // EPSILON_LINEAR_KRONECKER_PRODUCT_IMPL_H
