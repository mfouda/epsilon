
#include <unordered_set>

#include <glog/logging.h>

#include "epsilon/vector/block_matrix.h"

LinearMap& BlockMatrix::operator()(
    const std::string& row_key, const std::string& col_key) {
  return data_[col_key][row_key];
}

BlockMatrix BlockMatrix::Transpose() const {
  BlockMatrix transpose;
  for (const auto& col_iter : data_) {
    for (const auto& item_iter : col_iter.second) {
      transpose.InsertOrAdd(
          col_iter.first, item_iter.first, item_iter.second.Transpose());
    }
  }
  return transpose;
}

BlockMatrix BlockMatrix::Inverse() const {
  // Assumes a singleton
  // TODO(mwytock): Add other solvers, etc.

  CHECK_EQ(1, data_.size());
  auto iter = data_.begin();
  CHECK_EQ(1, iter->second.size());
  CHECK_EQ(iter->first, iter->second.begin()->first);

  const LinearMap& A = iter->second.begin()->second;
  CHECK_EQ(A.impl().m(), A.impl().n());

  BlockMatrix inv;
  inv.InsertOrAdd(iter->first, iter->first, A.Inverse());
  return inv;
}

BlockMatrix operator*(const BlockMatrix& A, const BlockMatrix& B) {
  BlockMatrix C;

  for (const auto& B_col_iter : B.data_) {
    for (const auto& B_iter : B_col_iter.second) {
      auto A_col_iter_ptr = A.data_.find(B_iter.first);
      if (A_col_iter_ptr == A.data_.end())
        continue;
      const auto& A_col_iter = *A_col_iter_ptr;

      for (const auto& A_iter : A_col_iter.second) {
        VLOG(3) << "C(" << A_iter.first << "," << B_col_iter.first << ") += "
                << "A(" << A_iter.first << "," << A_col_iter.first << ")*"
                << "B(" << B_iter.first << "," << B_col_iter.first << ")";
        VLOG(3) << A_iter.second.impl().m() << " x " << A_iter.second.impl().n();
        VLOG(3) << B_iter.second.impl().m() << " x " << B_iter.second.impl().n();
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
    LinearMap value) {
  auto res = data_[col_key].insert(std::make_pair(row_key, std::move(value)));
  if (!res.second) (res.first)->second += value;
}

int BlockMatrix::m() const {
  std::unordered_set<std::string> seen;
  int m = 0;
  for (const auto& col_iter : data_) {
    for (const auto& block_iter : col_iter.second) {
      auto seen_iter = seen.find(block_iter.first);
      if (seen_iter != seen.end())
        continue;
      m += block_iter.second.impl().m();
      seen.insert(block_iter.first);
    }
  }
  return m;
}

int BlockMatrix::n() const {
  int n = 0;
  for (auto col_iter : data_) {
    n += col_iter.second.begin()->second.impl().n();
  }
  return n;
}

const std::map<std::string, LinearMap>& BlockMatrix::col(
    const std::string& col_key) const {
  auto iter = data_.find(col_key);
  CHECK(iter != data_.end());
  return iter->second;
}

std::string BlockMatrix::DebugString() const {
  std::string retval = "";
  for (const auto& col_iter : data_) {
    for (const auto& block_iter : col_iter.second) {
      if (retval != "") retval += "\n";
      retval += "(" + block_iter.first + ", " + col_iter.first + ")\n";
      retval += block_iter.second.impl().DebugString();
    }
  }
  return retval;
}
