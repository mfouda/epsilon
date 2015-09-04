#ifndef DISTOPT_UTIL_PROBLEMS_H
#define DISTOPT_UTIL_PROBLEMS_H

#include "distopt/data.pb.h"
#include "distopt/expression.pb.h"
#include "distopt/util/vector.h"

// Silly norm 1 problem, min ||x||_1
MatrixProblem BuildNorm1(int n);
Expression BuildNorm1(int n, const std::string& var_id);

// Lasso variants (with different cone formulations)
MatrixProblem BuildLasso(
    const MatrixXd& A_lasso, const VectorXd& b_lasso, double lambda);
MatrixProblem BuildLassoCone(
    const MatrixXd& A_lasso, const VectorXd& b_lasso, double lambda);
MatrixProblem BuildLasso2(
    const MatrixXd& A_lasso, const VectorXd& b_lasso, double lambda);

Expression BuildLasso(
    const Expression& A, const Expression& b, double lambda,
    const std::string& var_id);

// ||Ax - b|| in various ways
MatrixProblem BuildLS(const MatrixXd& A, const VectorXd& b);
MatrixProblem BuildLS_Cone(const MatrixXd& A, const VectorXd& b);
MatrixProblem BuildLS_Cone_RowSplit(
  const MatrixXd& A, const VectorXd& b, int block_size);
MatrixProblem BuildMinAbsError(const MatrixXd& A, const VectorXd& b);

Expression BuildLS(
    const Expression& A, const Expression& b, const std::string& var_id);
Expression BuildLS_Cone(
    const Expression& A, const Expression& b, const std::string& var_id);
Expression BuildMinAbsError(
    const Expression& A, const Expression& b, const std::string& var_id);

// More complicated problems
MatrixProblem BuildSparseInverseCovarianceEstimation(
    const MatrixXd& S, double lambda);
MatrixProblem BuildRobustPCA(const MatrixXd& M, double mu);
MatrixProblem BuildProjLinearCone(
  const MatrixXd& A, const VectorXd& b, const VectorXd& x0, double t0);


// Random problem instances

Expression RandomDenseLasso(int m, int n, double p);

#endif  // DISTOPT_UTIL_PROBLEMS_H
