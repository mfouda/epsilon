#ifndef EPSILON_OPERATORS_AFFINE_H
#define EPSILON_OPERATORS_AFFINE_H

#include <Eigen/Dense>

#include "epsilon/expression/var_offset_map.h"
#include "epsilon/util/dynamic_matrix.h"

class Data;
class Expression;

// TODO(mwytock): Remove the above versions
// void BuildSparseAffineOperator(
//     const Expression& expr,
//     int n,
//     int offset,
//     std::vector<Eigen::Triplet<double>>* A_coeffs,
//     Eigen::VectorXd* B);

// // Simplfied version of above
// void BuildAffineOperator(
//     const Expression& expr,
//     int n,
//     int offset,
//     Eigen::MatrixXd* A,
//     Eigen::VectorXd* b);

void BuildAffineOperator(
    const Expression& expr,
    const VariableOffsetMap& offsets,
    DynamicMatrix* A,
    DynamicMatrix* b);


#endif  // EPSILON_OPERATORS_AFFINE_H
