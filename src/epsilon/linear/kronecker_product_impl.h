#ifndef EPSILON_LINEAR_KRONECKER_PRODUCT_IMPL_H
#define EPSILON_LINEAR_KRONECKER_PRODUCT_IMPL_H

#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/util/string.h"
#include "epsilon/vector/vector_util.h"

namespace linear_map {

class KroneckerProductImpl final : public LinearMapImpl {
 public:
  KroneckerProductImpl(LinearMap A, LinearMap B)
      : LinearMapImpl(KRONECKER_PRODUCT), A_(A), B_(B) {}

  int m() const override { return A_.impl().m()*B_.impl().m(); }
  int n() const override { return A_.impl().n()*B_.impl().n(); }

  std::string DebugString() const override {
    return StringPrintf("kronecker product\nA: %s\nB: %s",
                        A_.impl().DebugString().c_str(),
                        B_.impl().DebugString().c_str());
  }

  DenseMatrix AsDense() const override;
  DenseVector Apply(const DenseVector& x) const override;

  LinearMapImpl* Transpose() const override {
    return new KroneckerProductImpl(A_.Transpose(), B_.Transpose());
  }

  LinearMapImpl* Inverse() const override {
    return new KroneckerProductImpl(A_.Inverse(), B_.Inverse());
  }

  bool operator==(const LinearMapImpl& other) const override;

  // Scalar matrix API
  const LinearMap& A() const { return A_; }
  const LinearMap& B() const { return B_; }

 private:
  LinearMap A_, B_;
};

}  // namespace linear_map

#endif  // EPSILON_LINEAR_KRONECKER_PRODUCT_IMPL_H
