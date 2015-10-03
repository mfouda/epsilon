#ifndef EPSILON_OPERATORS_AFFINE_MATRIX_H
#define EPSILON_OPERATORS_AFFINE_MATRIX_H

#include <Eigen/Dense>

#include "epsilon/expression.pb.h"

namespace affine {

// When dealing with matrix-valued variables, many affine operators are better
// represented as AXB + C, rather than kron(B^T, A)vec(X) + vec(C).
struct MatrixOperator {
  Eigen::MatrixXd A, B, C;
};

// Build the affine operator representing the given expression. Currently this
// only supports expressions with a few operators (ADD, MULTIPLY, NEGATE) and a
// single expression variable.
MatrixOperator BuildMatrixOperator(const Expression& expr);

}  // namespace affine


#endif  // EPSILON_OPERATORS_AFFINE_MATRIX_H
