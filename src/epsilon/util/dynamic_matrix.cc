
#include "epsilon/util/dynamic_matrix.h"

void DynamicMatrix::RightMultiply(const DynamicMatrix& B) {
  VLOG(2) << "RightMultiply\nA:\n"
          << DebugString()
          << "\nB:\n"
          << B.DebugString();

  CHECK_EQ(cols(), B.rows());
  if (is_sparse_ && B.is_sparse_) {
    sparse_ = sparse_*B.sparse_;
  } else if (is_sparse_ && !B.is_sparse_) {
    dense_ = sparse_*B.dense_;
    sparse_ = SparseXd(0,0);
    is_sparse_ = false;
  } else if (!is_sparse_ && B.is_sparse_) {
    dense_ *= B.sparse_;
  } else if (!is_sparse_ && !B.is_sparse_) {
    dense_ *= B.dense_;
  } else {
    LOG(FATAL) << "Unknown A*B combination";
  }
}

void DynamicMatrix::Add(const DynamicMatrix& B, int i, int j) {
  const int m = B.rows();
  const int n = B.cols();

  CHECK_LE(i+m, rows());
  CHECK_LE(j+n, cols());

  if (is_sparse_ && B.is_sparse_) {
    CHECK(i == 0 && m == rows());
    sparse_.middleCols(j, n) += B.sparse_;

    // TODO(mwytock): Figure out bug in eigen sparse library?
    // for (int k = 0; k < B.sparse_.outerSize(); k++) {
    //   for (SparseXd::InnerIterator iter(B.sparse_, k); iter; ++iter) {
    //     sparse_.coeffRef(i+iter.row(), j+iter.col()) += iter.value();
    //   }
    // }
  } else if (is_sparse_ && !B.is_sparse_) {
    dense_ = static_cast<Eigen::MatrixXd>(sparse_);
    sparse_ = SparseXd(0,0);
    is_sparse_ = false;

    dense_.block(i, j, m, n) += B.dense_;
  } else if (!is_sparse_ && B.is_sparse_) {
    dense_.block(i, j, m, n) += B.sparse_;
  } else if (!is_sparse_ && !B.is_sparse_) {
    dense_.block(i, j, m, n) += B.dense_;
  } else {
    LOG(FATAL) << "Unknown A+B combination";
  }

}

void DynamicMatrix::ToVector() {
  if (is_sparse_) {
    LOG(FATAL) << "Not implemented";
  } else {
    dense_ = Eigen::Map<Eigen::MatrixXd>(dense_.data(), rows()*cols(), 1);
  }
}

void DynamicMatrix::ToMatrix(int m, int n) {
  CHECK_EQ(cols(), 1);
  CHECK_EQ(rows(), m*n);

  if (is_sparse_) {
    LOG(FATAL) << "Not implemented";
  } else {
    dense_ = Eigen::Map<Eigen::MatrixXd>(dense_.data(), m, n);
  }
}

DynamicMatrix DynamicMatrix::GetColumns(int j, int n) {
  if (is_sparse_) {
    return DynamicMatrix::FromSparse(sparse_.middleCols(j, n));
  } else {
    return DynamicMatrix::FromDense(dense_.middleCols(j, n));
  }
}

DynamicMatrix DynamicMatrix::GetRows(int i, int m) {
  if (is_sparse_) {
    return DynamicMatrix::FromSparse(sparse_.middleRows(i, m));
  } else {
    return DynamicMatrix::FromDense(dense_.middleRows(i, m));
  }
}
