
#include "epsilon/vector/block_matrix.h"

class LLTDenseSolver final : public BlockMatrix::Solver {
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> DenseMatrix;

 public:
  LLTDenseSolver(const BlockMatrix& matrix, double lambda) {
    // Build/factor dense representation
  }

  BlockVector solve(const BlockVector& b) override {
    return BlockVector();
  }

private:
  Eigen::LLT<DenseMatrix> solver_;
};

MatrixVariant& BlockMatrix::operator()(
    const std::string& row_key, const std::string& col_key) {
  return data_[col_key][row_key];
}

BlockMatrix BlockMatrix::transpose() const {
  BlockMatrix transpose;
  for (auto col_iter : data_) {
    for (auto item_iter : col_iter.second) {
      transpose(col_iter.first, item_iter.first) =
          item_iter.second.transpose();
    }
  }
  return transpose;
}

std::unique_ptr<BlockMatrix::Solver> BlockMatrix::inv(double lambda) const {
  // TODO(mwytock): Add other solvers, etc.
  return std::unique_ptr<BlockMatrix::Solver>(
      new LLTDenseSolver(*this, lambda));
}

BlockMatrix operator*(const BlockMatrix& A, const BlockMatrix& B) {
  BlockMatrix C;

  for (auto B_col_iter : B.data_) {
    for (auto B_iter : B_col_iter.second) {
      auto A_col_iter_ptr = A.data_.find(B_iter.first);
      if (A_col_iter_ptr == A.data_.end())
        continue;
      const auto& A_col_iter = *A_col_iter_ptr;

      for (auto A_iter : A_col_iter.second) {
        VLOG(3) << "C(" << A_iter.first << "," << B_col_iter.first << ") += "
                << "A(" << A_iter.first << "," << A_col_iter.first << ")*"
                << "B(" << B_iter.first << "," << B_col_iter.first << ")";
        VLOG(3) << A_iter.second.rows() << " x " << A_iter.second.cols();
        VLOG(3) << B_iter.second.rows() << " x " << B_iter.second.cols();
        C.InsertOrAdd(
            A_iter.first, B_col_iter.first, A_iter.second*B_iter.second);
      }
    }
  }

  return C;
}

BlockVector operator*(const BlockMatrix& A, const BlockVector& x) {
  BlockVector y;
  for (auto x_iter : x.data_) {
    auto col_iter = A.data_.find(x_iter.first);
    if (col_iter == A.data_.end())
      continue;

    for (auto block_iter : col_iter->second)
      y.InsertOrAdd(block_iter.first, block_iter.second*x_iter.second);
  }
  return y;
}

void BlockMatrix::InsertOrAdd(
    const std::string& row_key,
    const std::string& col_key,
    MatrixVariant value) {
  auto res = data_[col_key].insert(std::make_pair(row_key, value));
  if (!res.second) (res.first)->second += value;
}
