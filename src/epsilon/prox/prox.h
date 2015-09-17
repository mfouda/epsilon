#ifndef EPSILON_OPERATORS_PROX_H
#define EPSILON_OPERATORS_PROX_H

#include <vector>
#include <memory>

#include "epsilon/expression.pb.h"
#include "epsilon/expression/var_offset_map.h"
#include "epsilon/vector/vector_operator.h"

std::unique_ptr<VectorOperator> CreateProxOperator(
    double lambda,
    const Expression& f_expr,
    const VariableOffsetMap& var_map);

template<class T>
bool RegisterProxOperator(const std::string& id);
#define REGISTER_PROX_OPERATOR(T) bool registered_##T = RegisterProxOperator<T>(#T)

#endif  // EPSILON_OPERATORS_PROX_H
