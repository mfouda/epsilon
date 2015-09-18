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

// alpha*x + b
void GetScalarAffineOperator(
    const Expression& expr,
    const VariableOffsetMap& var_map,
    double* alpha,
    Eigen::VectorXd* b);

// a.*x + b
void GetDiagonalAffineOperator(
    const Expression& expr,
    const VariableOffsetMap& var_map,
    Eigen::VectorXd* a,
    Eigen::VectorXd* b);

// TODO(mwytock): Consider how to represent this better
SparseXd GetSparseAffineOperator(
    const Expression& expr,
    const VariableOffsetMap& var_map);


#endif  // EPSILON_OPERATORS_AFFINE_H
