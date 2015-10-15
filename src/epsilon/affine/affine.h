#ifndef EPSILON_OPERATORS_AFFINE_H
#define EPSILON_OPERATORS_AFFINE_H

#include <Eigen/Dense>

#include "epsilon/expression/var_offset_map.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/matrix_variant.h"

class BlockMatrix;
class BlockVector;
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

// Get orthogonal projection matrix so that P*x reorders x variables from the
// offsets in a to those in b.
SparseXd GetProjection(const VariableOffsetMap& a, const VariableOffsetMap& b);


namespace affine {

MatrixVariant BuildLinearOperator(const Expression& expr);

}  // namespace affine

#endif  // EPSILON_OPERATORS_AFFINE_H
