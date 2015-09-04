#ifndef EPSILON_OPERATORS_VECTOR_OPERATOR_H
#define EPSILON_OPERATORS_VECTOR_OPERATOR_H

#include <Eigen/Dense>

class VectorOperator {
 public:
  virtual void Init() {}
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& x) = 0;
};

#endif  // EPSILON_OPERATORS_VECTOR_OPERATOR_H
