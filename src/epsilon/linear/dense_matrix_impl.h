#ifndef EPSILON_LINEAR_DENSE_MATRIX_IMPL_H
#define EPSILON_LINEAR_DENSE_MATRIX_IMPL_H

#include <memory>

#include <glog/logging.h>

#include "epsilon/linear/linear_map.h"

namespace linear_map {


class DenseMatrixImpl final : public LinearMapImpl {
 public:
  struct Data {
    std::unique_ptr<Scalar[]> data;
  };

  // Makes a copy of A's data
  // TODO(mwytock): Eventually this constructor should be removed in favor of
  // the new explicit constructor passing a shared_ptr to data.
  DenseMatrixImpl(const DenseMatrix& A);

  // Shares a pointer to data
  DenseMatrixImpl(
      int m, int n, std::shared_ptr<Data> data_ptr, char trans) :
      LinearMapImpl(DENSE_MATRIX),
      m_(m), n_(n),
      data_ptr_(data_ptr),
      trans_(trans) {}

  int m() const override { return trans_ == 'N' ? m_ : n_; }
  int n() const override { return trans_ == 'N' ? n_ : m_; }
  std::string DebugString() const override;
  DenseMatrix AsDense() const override {
    DenseMatrix A = Eigen::Map<DenseMatrix>(data(), m_, n_);
    return trans_ == 'N' ? A : static_cast<DenseMatrix>(A.transpose());
  }
  DenseVector Apply(const DenseVector& x) const override;

  LinearMapImpl* Transpose() const override {
    return new DenseMatrixImpl(m_, n_, data_ptr_, trans_ == 'T' ? 'N' : 'T');
  }

  LinearMapImpl* Inverse() const override;

  bool operator==(const LinearMapImpl& other) const override;

  // Dense matrix API

  // Direct access for lapack calls
  Scalar* data() const { return data_ptr_->data.get(); }
  char* trans() const { return const_cast<char*>(&trans_); }

 private:

  int m_, n_;
  std::shared_ptr<Data> data_ptr_;
  char trans_;
};

}  // namespace linear_map

#endif  // EPSILON_LINEAR_DENSE_MATRIX_IMPL_H
