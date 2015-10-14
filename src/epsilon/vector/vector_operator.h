#ifndef EPSILON_OPERATORS_VECTOR_OPERATOR_H
#define EPSILON_OPERATORS_VECTOR_OPERATOR_H

#include <Eigen/Dense>

#include "epsilon/vector/block_vector.h"

class VectorOperator {
 public:
  virtual void Init() {}
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& x) = 0;
};

class BlockVectorOperator {
 public:
  virtual void Init() {}
  virtual BlockVector Apply(const BlockVector& x) = 0;
};

#endif  // EPSILON_OPERATORS_VECTOR_OPERATOR_H
