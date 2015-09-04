#ifndef DISTOPT_OPERATORS_PROX_H
#define DISTOPT_OPERATORS_PROX_H

#include <vector>
#include <memory>

#include "distopt/prox.pb.h"
#include "distopt/operators/vector_operator.h"

std::unique_ptr<VectorOperator> CreateProxOperator(
    const ProxFunction& f,
    double lambda,
    int n);

#endif  // DISTOPT_OPERATORS_PROX_H
