#ifndef EPSILON_OPERATORS_AFFINE_H
#define EPSILON_OPERATORS_AFFINE_H

#include <Eigen/Dense>

#include "epsilon/expression/var_offset_map.h"
#include "epsilon/vector/block_matrix.h"
#include "epsilon/vector/vector_util.h"

class BlockMatrix;
class BlockVector;
class Data;
class Expression;

struct AffineOperator {
  BlockMatrix A;
  BlockVector b;
};

namespace affine {

std::string constraint_key(int i);
std::string arg_key(int i);

// Convenience function, calls BuildConstant() and BuildLinearMap() on the
// expressions and puts the result in row_key in the BlockMatrix
void BuildAffineOperator(
    const Expression& expr,
    const DataMap& data_map,
    const std::string& row_key,
    BlockMatrix* A,
    BlockVector* b);

// Convenience methods for "simple" affine operators
std::string GetSingleVariableKey(const AffineOperator& op);
double GetScalar(const AffineOperator& op);
Eigen::VectorXd GetDiagonal(const AffineOperator& op);
Eigen::VectorXd GetConstant(const AffineOperator& op);

BlockVector GetLinear(const AffineOperator& op);


}  // namespace affine

#endif  // EPSILON_OPERATORS_AFFINE_H
