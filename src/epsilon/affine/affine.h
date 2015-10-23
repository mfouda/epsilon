#ifndef EPSILON_OPERATORS_AFFINE_H
#define EPSILON_OPERATORS_AFFINE_H

#include <Eigen/Dense>

#include "epsilon/expression/var_offset_map.h"
#include "epsilon/vector/block_matrix.h"

class BlockMatrix;
class BlockVector;
class Data;
class Expression;

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
