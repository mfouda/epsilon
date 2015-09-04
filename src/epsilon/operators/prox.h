#ifndef EPSILON_OPERATORS_PROX_H
#define EPSILON_OPERATORS_PROX_H

#include <vector>
#include <memory>

#include "epsilon/prox.pb.h"
#include "epsilon/operators/vector_operator.h"

std::unique_ptr<VectorOperator> CreateProxOperator(
    const ProxFunction& f,
    double lambda,
    int n);

#endif  // EPSILON_OPERATORS_PROX_H
