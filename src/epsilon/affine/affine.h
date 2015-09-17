#ifndef EPSILON_OPERATORS_AFFINE_H
#define EPSILON_OPERATORS_AFFINE_H

#include <Eigen/Dense>

#include "epsilon/expression/var_offset_map.h"
#include "epsilon/vector/dynamic_matrix.h"

class Data;
class Expression;

void BuildAffineOperator(
    const Expression& expr,
    const VariableOffsetMap& offsets,
    DynamicMatrix* A,
    DynamicMatrix* b);

// Utility functions of the above, sparse A and b = 0
SparseXd GetSparseAffineOperator(
    const Expression& expr,
    const VariableOffsetMap& var_map);

#endif  // EPSILON_OPERATORS_AFFINE_H
