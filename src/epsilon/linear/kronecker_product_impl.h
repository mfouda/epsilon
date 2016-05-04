#ifndef EPSILON_LINEAR_KRONECKER_PRODUCT_IMPL_H
#define EPSILON_LINEAR_KRONECKER_PRODUCT_IMPL_H

#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/util/string.h"
#include "epsilon/vector/vector_util.h"

namespace linear_map {

class KroneckerProductImpl final : public LinearMapImpl {
 public:
  KroneckerProductImpl(LinearMapImpl* A, LinearMapImpl* B)
      : LinearMapImpl(KRONECKER_PRODUCT), A_(A), B_(B) {}

  int m() const override { return A_->m()*B_->m(); }
  int n() const override { return A_->n()*B_->n(); }

  std::string DebugString() const override {
    return StringPrintf("kronecker product\nA: %s\nB: %s",
                        A_->DebugString().c_str(),
                        B_->DebugString().c_str());
  }

  DenseMatrix AsDense() const override;
  DenseVector Apply(const DenseVector& x) const override;
  DenseMatrix ApplyMatrix(const DenseMatrix& X) const override {
    LOG(FATAL) << "Not implemented";
  }

  LinearMapImpl* Transpose() const override {
    return new KroneckerProductImpl(A_->Transpose(), B_->Transpose());
  }

  LinearMapImpl* Inverse() const override {
    return new KroneckerProductImpl(A_->Inverse(), B_->Inverse());
  }

  LinearMapImpl* Clone() const override {
    return new KroneckerProductImpl(A_->Clone(), B_->Clone());
  }

  bool operator==(const LinearMapImpl& other) const override;

  // Kronecker product specific methods
  const LinearMapImpl& A() const { return *A_; }
  const LinearMapImpl& B() const { return *B_; }
  SparseMatrix AsSparse() const;

 private:
  std::unique_ptr<LinearMapImpl> A_, B_;
};

}  // namespace linear_map

#endif  // EPSILON_LINEAR_KRONECKER_PRODUCT_IMPL_H
