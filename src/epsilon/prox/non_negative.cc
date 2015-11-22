#include "epsilon/prox/non_negative.h"

Eigen::VectorXd NonNegativeProx::ApplyElementwise(const Eigen::VectorXd& v) {
    return v.cwiseMax(0);
}
REGISTER_PROX_OPERATOR(NON_NEGATIVE, NonNegativeProx);
