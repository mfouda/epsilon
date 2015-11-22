#ifndef EPSILON_PROX_NON_NEGATIVE_H
#define EPSILON_PROX_NON_NEGATIVE_H

#include "epsilon/prox/elementwise.h"

// I(x >= 0)
class NonNegativeProx final : public ElementwiseProx {
public:
  Eigen::VectorXd ApplyElementwise(const Eigen::VectorXd& v) override;
};

#endif  // EPSILON_PROX_NON_NEGATIVE_H
