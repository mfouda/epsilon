#ifndef EPSILON_UTIL_DYNAMIC_MATRIX_H
#define EPSILON_UTIL_DYNAMIC_MATRIX_H

#include <glog/logging.h>

#include "epsilon/util/vector.h"
#include "epsilon/util/string.h"

// Can hold either a sparse or dense matrix
class DynamicMatrix {
 public:
  DynamicMatrix() : is_sparse_(true) {}
  DynamicMatrix(int m, int n)
      : is_sparse_(true), sparse_(m, n) {}

  DynamicMatrix(const DynamicMatrix& B) {
    if (B.is_sparse_) {
      is_sparse_ = true;
      sparse_ = B.sparse_;
    } else {
      is_sparse_ = false;
      dense_ = B.dense_;
    }
  }

  DynamicMatrix(DynamicMatrix&& B) {
    if (B.is_sparse_) {
      is_sparse_ = true;
      sparse_ = std::move(B.sparse_);
    } else {
      is_sparse_ = false;
      dense_ = std::move(B.dense_);
    }
  }

  ~DynamicMatrix() {}

  DynamicMatrix& operator= (const DynamicMatrix& B) {
    DynamicMatrix A(B);
    *this = std::move(A);
    return *this;
  }

  DynamicMatrix& operator= (DynamicMatrix&& B) {
    if (B.is_sparse_) {
      is_sparse_ = true;
      std::swap(sparse_, B.sparse_);
    } else {
      is_sparse_ = false;
      std::swap(dense_, B.dense_);
    }
    return *this;
  }


  // TODO(mwytock): This form deprecated, remove
  void Apply(const VectorXd& x, VectorXd* y) {
    *y = Apply(x);
  }

  VectorXd Apply(const VectorXd& x) {
    if (is_sparse()) {
      CHECK_EQ(sparse_.cols(), x.rows());
      return sparse_ * x;
    }

    CHECK_EQ(dense_.cols(), x.rows());
    return dense_ * x;
  }

  void ApplyTranspose(const VectorXd& x, VectorXd* y) {
    if (is_sparse()) {
      *y = sparse_.transpose() * x;
    } else {
      *y = dense_.transpose() * x;
    }
  }

  MatrixXd AsDense() const {
    if (is_sparse()) {
      MatrixXd dense = sparse_;
      return dense;
    }

    // Dense
    return dense_;
  }

  SparseXd AsSparse() const {
    CHECK(is_sparse_);
    return sparse_;
  }

  // Form A = A*B with A == this
  void RightMultiply(const DynamicMatrix& B);
  void Add(const DynamicMatrix& B, int i=0, int j=0);

  // Reshape operators
  void ToVector();
  void ToMatrix(int m, int n);

  // Subset operators
  DynamicMatrix GetColumns(int j, int n);
  DynamicMatrix GetRows(int i, int m);

  int rows() const { return is_sparse_ ? sparse_.rows() : dense_.rows(); }
  int cols() const { return is_sparse_ ? sparse_.cols() : dense_.cols(); }

  bool is_sparse() const {
    return is_sparse_;
  }

  bool is_zero() const {
    if (is_sparse_)
      return sparse_.nonZeros() == 0;
    else
      return dense_.isZero();
  }

  const MatrixXd& dense() const {
    CHECK(!is_sparse_);
    return dense_;
  }

  const SparseXd& sparse() const {
    CHECK(is_sparse_);
    return sparse_;
  }

  static DynamicMatrix FromSparse(SparseXd sparse) {
    DynamicMatrix dm;
    dm.is_sparse_ = true;
    dm.sparse_ = sparse;
    return dm;
  }

  static DynamicMatrix FromDense(MatrixXd dense) {
    DynamicMatrix dm;
    dm.is_sparse_ = false;
    dm.dense_ = dense;
    return dm;
  }

  static DynamicMatrix Zero(int m, int n) {
    return FromSparse(SparseXd(m, n));
  }

  static DynamicMatrix Identity(int n) {
    return FromSparse(SparseIdentity(n));
  }

  std::string DebugString() const {
    std::string str = StringPrintf(
        "%s (%d, %d)\n", is_sparse_ ? "sparse" : "dense", rows(), cols());

    if (is_sparse_)
      return str + SparseMatrixDebugString(sparse_);
    else
      return str + MatrixDebugString(dense_);
  }

 private:
  bool is_sparse_;
  SparseXd sparse_;
  MatrixXd dense_;
};

#endif  // EPSILON_UTIL_DYNAMIC_MATRIX_H
