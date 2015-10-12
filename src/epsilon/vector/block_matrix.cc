
#include "epsilon/vector/block_matrix.h"

class SingletonSolver : public BlockMatrix::Solver {
 public:
  SingletonSolver(
      std::string key,
      std::unique_ptr<MatrixVariant::Solver> solver)
      : key_(key),
        solver_(std::move(solver)) {}

  BlockVector solve(const BlockVector& b) const {
    BlockVector x;
    x(key_) = solver_->solve(b(key_));
    return x;
  }

 private:
  std::string key_;
  std::unique_ptr<MatrixVariant::Solver> solver_;
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

std::unique_ptr<BlockMatrix::Solver> BlockMatrix::inv() const {
  // Assumes a singleton, symmetric matrix

  CHECK_EQ(1, data_.size());
  auto iter = data_.begin();
  CHECK_EQ(1, iter->second.size());
  CHECK_EQ(iter->first, iter->second.begin()->first);
  return std::unique_ptr<BlockMatrix::Solver>(
      new SingletonSolver(iter->first, iter->second.begin()->second.inv()));
  // TODO(mwytock): Add other solvers, etc.

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
