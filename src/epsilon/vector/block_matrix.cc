
#include "epsilon/vector/block_matrix.h"

MatrixVariant& BlockMatrix::operator()(
    const std::string& row_key, const std::string& col_key) {
  return data_[col_key][row_key];
}

BlockMatrix& BlockMatrix::operator*=(const BlockMatrix& rhs) {
  return *this;
}

BlockMatrix BlockMatrix::transpose() const {
  return *this;
}

BlockMatrix::Solver BlockMatrix::inv(double lambda) const {
  return BlockMatrix::Solver();
}

BlockMatrix operator*(BlockMatrix lhs, const BlockMatrix& rhs) {
  lhs *= rhs;
  return lhs;
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
