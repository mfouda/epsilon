// Computes the LDL factorization of a block matrix composed of LinearMaps.

#ifndef EPSILON_VECTOR_BLOCK_CHOLESKY_H
#define EPSILON_VECTOR_BLOCK_CHOLESKY_H

#include "epsilon/vector/block_matrix.h"

class BlockCholesky {
 public:
  void Compute(BlockMatrix A);
  BlockVector Solve(const BlockVector& b);

 private:
  std::vector<std::string> p_, p_rev_;
  BlockMatrix D_inv_, L_;
};

#endif  // EPSILON_VECTOR_BLOCK_CHOLESKY_H
