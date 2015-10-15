
#include <unordered_set>

#include <glog/logging.h>

#include "epsilon/vector/block_matrix.h"

class SingletonSolver : public BlockMatrix::Solver {
 public:
  SingletonSolver(
      std::string key,
      int n,
      std::unique_ptr<MatrixVariant::Solver> solver)
      : key_(key),
        n_(n),
        solver_(std::move(solver)) {}

  BlockVector solve(const BlockVector& b) const {
    BlockVector x;
    x(key_) = solver_->solve(
        b.has_key(key_) ? b(key_) : MatrixVariant::DenseVector::Zero(n_));
    return x;
  }

 private:
  std::string key_;
  int n_;
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
  // TODO(mwytock): Add other solvers, etc.

  CHECK_EQ(1, data_.size());
  auto iter = data_.begin();
  CHECK_EQ(1, iter->second.size());
  CHECK_EQ(iter->first, iter->second.begin()->first);

  const MatrixVariant& A = iter->second.begin()->second;
  CHECK_EQ(A.rows(), A.cols());

  return std::unique_ptr<BlockMatrix::Solver>(
      new SingletonSolver(iter->first, A.rows(), A.inv()));

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
  VLOG(3) << "block matrix-vector product";
  BlockVector y;
  for (const auto& x_iter : x.data_) {
    auto col_iter = A.data_.find(x_iter.first);
    if (col_iter == A.data_.end())
      continue;

    for (const auto& block_iter : col_iter->second)
      y.InsertOrAdd(block_iter.first, block_iter.second*x_iter.second);
  }
  VLOG(3) << "block matrix-vector product done";
  return y;
}

void BlockMatrix::InsertOrAdd(
    const std::string& row_key,
    const std::string& col_key,
    MatrixVariant value) {
  auto res = data_[col_key].insert(std::make_pair(row_key, value));
  if (!res.second) (res.first)->second += value;
}

int BlockMatrix::rows() const {
  std::unordered_set<std::string> seen;
  int m = 0;
  for (auto col_iter : data_) {
    for (auto block_iter : col_iter.second) {
      auto seen_iter = seen.find(block_iter.first);
      if (seen_iter != seen.end())
        continue;
      m += block_iter.second.rows();
      seen.insert(block_iter.first);
    }
  }
  return m;
}

int BlockMatrix::cols() const {
  int n = 0;
  for (auto col_iter : data_) {
    n += col_iter.second.begin()->second.cols();
  }
  return n;
}

const std::map<std::string, MatrixVariant>& BlockMatrix::col(
    const std::string& col_key) const {
  auto iter = data_.find(col_key);
  CHECK(iter != data_.end());
  return iter->second;
}

std::string BlockMatrix::DebugString() const {
  std::string retval = "";
  for (auto col_iter : data_) {
    for (auto block_iter : col_iter.second) {
      if (retval != "") retval += "\n";
      retval += "(" + block_iter.first + ", " + col_iter.first + ")\n";
      retval += block_iter.second.DebugString();

    }
  }
  return retval;
}
