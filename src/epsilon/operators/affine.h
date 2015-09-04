#ifndef EPSILON_OPERATORS_AFFINE_H
#define EPSILON_OPERATORS_AFFINE_H

#include <string>
#include <unordered_map>

#include <Eigen/Dense>

#include "epsilon/util/dynamic_matrix.h"
#include "epsilon/util/vector.h"

class Data;
class Expression;

class OperatorImpl {
 public:
  virtual void Apply(const Eigen::VectorXd& x, Eigen::VectorXd* y) = 0;

  // May be slow, should not be called for large matrices
  virtual void ToMatrix(Eigen::MatrixXd* A) = 0;
};

OperatorImpl* BuildLinearExpressionOperator(
    const Expression& expression, bool transpose);

OperatorImpl* BuildLinearProjectionOperator(const Eigen::MatrixXd& A);
OperatorImpl* BuildLinearProjectionOperator(const SparseXd& A);

// TODO(mwytock): Remove the above versions
void BuildSparseAffineOperator(
    const Expression& expr,
    int n,
    int offset,
    std::vector<Eigen::Triplet<double>>* A_coeffs,
    Eigen::VectorXd* B);

// Simplfied version of above
void BuildAffineOperator(
    const Expression& expr,
    int n,
    int offset,
    Eigen::MatrixXd* A,
    Eigen::VectorXd* b);

void BuildAffineOperator(
    const Expression& expr,
    DynamicMatrix* A,
    DynamicMatrix* b);


#endif  // EPSILON_OPERATORS_AFFINE_H
