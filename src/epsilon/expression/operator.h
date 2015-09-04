#ifndef EXPRESSION_OPERATOR_H
#define EXPRESSION_OPERATOR_H

#include <string>
#include <unordered_map>

#include <Eigen/Dense>

#include "distopt/util/dynamic_matrix.h"
#include "distopt/util/vector.h"

class Data;
class Expression;
class Operator;

using Eigen::VectorXd;
using Eigen::MatrixXd;

class OperatorImpl {
 public:
  virtual void Apply(const VectorXd& x, VectorXd* y) = 0;

  // May be slow, should not be called for large matrices
  virtual void ToMatrix(MatrixXd* A) = 0;
};

OperatorImpl* BuildLinearExpressionOperator(
    const Expression& expression, bool transpose);

OperatorImpl* BuildLinearProjectionOperator(const MatrixXd& A);
OperatorImpl* BuildLinearProjectionOperator(const SparseXd& A);

// TODO(mwytock): Remove the above versions
void BuildSparseAffineOperator(
    const Expression& expr,
    int n,
    int offset,
    std::vector<Eigen::Triplet<double>>* A_coeffs,
    VectorXd* B);

// Simplfied version of above
void BuildAffineOperator(
    const Expression& expr,
    int n,
    int offset,
    MatrixXd* A,
    VectorXd* b);

void BuildAffineOperator(
    const Expression& expr,
    DynamicMatrix* A,
    DynamicMatrix* b);

std::string OperatorKey(const Operator& op);


#endif  // EXPRESSION_OPERATOR_H
