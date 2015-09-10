#ifndef EPSILON_OPERATORS_PROX_H
#define EPSILON_OPERATORS_PROX_H

#include <vector>
#include <memory>

#include "epsilon/expression.pb.h"
#include "epsilon/operators/vector_operator.h"

std::unique_ptr<VectorOperator> CreateProxOperator(
    const Expression& expr, double lambda);

#endif  // EPSILON_OPERATORS_PROX_H
