
#include "epsilon/vector/block_cholesky.h"

int ComputeFill(const BlockMatrix& A, const std::string& key) {
  // TODO(mwytock): Implement this;
  return 0;
}

// Choose the next key for the Cholesky decomposition, minimizing fill in.
std::string NextKey(const BlockMatrix& A) {
  std::string best_key;
  int best_fill = A.m()*A.n()+1;
  for (const std::string& key : A.col_keys()) {
    int fill = ComputeFill(A, key);
    if (ComputeFill(A, key) < best_fill) {
      best_key = key;
      best_fill = fill;
    }
  }
  return best_key;
}

// Remove the rows/columns corresponding to key
BlockMatrix RemoveKey(BlockMatrix* A, const std::string& key) {
  BlockMatrix V;
  for (const auto& iter : A->col(key)) {
    A->Remove(iter.first, key);
    A->Remove(key, iter.first);
    V(iter.first, key) = iter.second;
  }
  return V;
}

// Solve the system Lx = b where L is lower triangular when traversed in the
// ordered specified by keys.
BlockVector BackSub(
    const BlockMatrix& L,
    const std::vector<std::string>& keys,
    BlockVector b) {
  const int n = keys.size();
  for (int idx_i = 0; idx_i < n; idx_i++) {
    const std::string& i = keys[idx_i];
    for (int idx_j = idx_i + 1; idx_j < n; idx_j++) {
      const std::string& j = keys[idx_j];
      b(j) -= L(j,i)*b(i);
    }
  }
  return b;
}

void BlockCholesky::Compute(BlockMatrix A) {
  const int n_cols = A.col_keys().size();

  for (int i = 0; i < n_cols; i++) {
    std::string key = NextKey(A);
    BlockMatrix Di_inv;
    Di_inv(key, key) = A(key, key).Inverse();
    BlockMatrix V = RemoveKey(&A, key);

    L_ = L_ + V*Di_inv;
    D_inv_ = D_inv_ + Di_inv;
    A = A - V*Di_inv*V.Transpose();
    p_.push_back(key);
  }
  std::reverse_copy(p_.begin(), p_.end(), p_rev_.begin());
}

BlockVector BlockCholesky::Solve(const BlockVector& b) {
  return BackSub(L_, p_rev_, D_inv_*BackSub(L_, p_, b));
}
