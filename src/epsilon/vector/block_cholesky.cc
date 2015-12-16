
#include <limits>

#include "epsilon/vector/block_cholesky.h"

const uint64_t kFillMax = std::numeric_limits<uint64_t>::max();

// Compute the maximum number of nonzeros in VD^{-1}V^T where V is the
// column of A corresponding to key and D is the diagonal block corresponding to
// key.
uint64_t ComputeFill(const BlockMatrix& A, const std::string& k) {
  VLOG(2) << "ComputeFill: " << k;
  std::set<std::string> keys;
  bool has_diagonal = false;
  for (const auto& iter : A.col(k)) {
    if (k == iter.first) {
      has_diagonal = true;
    } else {
      keys.insert(iter.first);
    }
  }

  if (!has_diagonal)
    return kFillMax;

  uint64_t fill = 0;
  for (const std::string& i : keys) {
    linear_map::ImplType Aik_type = linear_map::ComputeType(
        linear_map::MULTIPLY,
        A(i, k).impl().type(),
        A(k, k).impl().type());

    for (const std::string& j : keys) {
      linear_map::ImplType type = linear_map::ComputeType(
          linear_map::MULTIPLY, Aik_type, A(j, k).impl().type());

      fill += linear_map::Nonzeros(
          type, A(i, k).impl().m(), A(j, k).impl().m());
    }
  }
  return fill;
}

// Choose the next key for the Cholesky decomposition, minimizing fill in.
std::string NextKey(const BlockMatrix& A) {
  std::string best_key;
  uint64_t best_fill = kFillMax;
  for (const std::string& key : A.col_keys()) {
    uint64_t fill = ComputeFill(A, key);
    if (fill < best_fill) {
      best_key = key;
      best_fill = fill;
    }
  }
  CHECK_NE(kFillMax, best_fill);
  VLOG(2) << "key: " << best_key << ", fill: " << best_fill;
  return best_key;
}

// Remove the row/column corresponding to the given key and return a BlockMatrix
// corresponding to the column but without the diagonal element.
BlockMatrix RemoveKey(BlockMatrix* A, const std::string& key) {
  BlockMatrix V;
  std::vector<std::pair<std::string, std::string>> to_remove;
  for (const auto& iter : A->col(key)) {
    to_remove.push_back(std::make_pair(iter.first, key));
    if (key != iter.first) {
      to_remove.push_back(std::make_pair(key, iter.first));
      V(iter.first, key) = iter.second;
    }
  }

  for (const auto& iter : to_remove) {
    A->Remove(iter.first, iter.second);
  }
  return V;
}

// Solve the system Lx = b where L is lower triangular
BlockVector ForwardSub(
    const BlockMatrix& L,
    const std::vector<std::string>& keys,
    BlockVector b) {
  for (auto j = keys.begin(); j != keys.end(); ++j) {
    if (b.has_key(*j)) {
      Eigen::VectorXd neg_bj = -1*b(*j);
      for (auto i = j + 1; i != keys.end(); ++i) {
        if (L.has_key(*i, *j))
          b.InsertOrAdd(*i, L(*i,*j)*neg_bj);
      }
    }
  }
  return b;
}

// Solve the system L'x = b where L is lower triangular
BlockVector BackSub(
    const BlockMatrix& L,
    const std::vector<std::string>& keys,
    BlockVector b) {
  for (auto j = keys.rbegin(); j != keys.rend(); j++) {
    if (b.has_key(*j)) {
      Eigen::VectorXd neg_bj = -1*b(*j);
      for (auto i = j + 1; i != keys.rend(); i++) {
        if (L.has_key(*i, *j))
          b.InsertOrAdd(*i, L(*i,*j)*neg_bj);
      }
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
  LT_ = L_.Transpose();
}

BlockVector BlockCholesky::Solve(const BlockVector& b) {
  return BackSub(LT_, p_, D_inv_*ForwardSub(L_, p_, b));
}
