#ifndef EPSILON_OPERATORS_AFFINE_H
#define EPSILON_OPERATORS_AFFINE_H

#include <Eigen/Dense>

#include "epsilon/expression/var_offset_map.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/vector/block_matrix.h"
#include "epsilon/vector/dynamic_matrix.h"

class BlockMatrix;
class BlockVector;
class Data;
class Expression;

// TODO(mwytock): Delete the below
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

// Convenience function, calls BuildConstant() and BuildLinearMap() on the
// expressions and puts the result in row_key in the BlockMatrix
void BuildAffineOperator(
    const Expression& expr,
    const std::string& row_key,
    BlockMatrix* A,
    BlockVector* b);

}  // namespace affine

#endif  // EPSILON_OPERATORS_AFFINE_H
