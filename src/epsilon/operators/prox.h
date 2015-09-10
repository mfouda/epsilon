#ifndef EPSILON_OPERATORS_PROX_H
#define EPSILON_OPERATORS_PROX_H

#include <vector>
#include <memory>

#include "epsilon/expression.pb.h"
#include "epsilon/expression/var_offset_map.h"
#include "epsilon/operators/vector_operator.h"

std::unique_ptr<VectorOperator> CreateProxOperator(
    double lambda,
    const Expression& f_expr,
    const VariableOffsetMap& var_map);


#endif  // EPSILON_OPERATORS_PROX_H
