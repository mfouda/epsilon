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

  int m() const override { return m_; }
  int n() const override { return n_; }
  std::string DebugString() const override;
  DenseMatrix AsDense() const override;
  DenseVector Apply(const DenseVector& x) const override;

  LinearMapImpl* Transpose() const override {
    return new DenseMatrixImpl(n_, m_, data_ptr_, trans_ == 'Y' ? 'N' : 'Y');
  }

  LinearMapImpl* Inverse() const override;

  bool operator==(const LinearMapImpl& other) const override;

  // Dense matrix API

  // Eigen representation for multipying other types
  Eigen::Map<const DenseMatrix> dense() const {
    return Eigen::Map<const DenseMatrix>(data(), m(), n());
  }

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
