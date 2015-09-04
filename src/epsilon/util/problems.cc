#include "distopt/util/problems.h"

#include <Eigen/SparseCore>
#include <glog/logging.h>

#include "distopt/expression/expression.h"

using Eigen::Map;

const std::string kVariablePrefix = "var";

// Construct lasso problem in epigraph form
//
// minimize_x 1/2*||Ax - b||^2 + lambda*||x||_1
// is equivalent to:
//
// minimize_r,x,t,s_r,s_x,s_t 1/2*r'r + lambda*t
// subject to
//   Ax - r + s_r = b
//   -x + s_x = 0
//   -t + s_t = 0
//   (s_x, s_t) in {(x, t) : ||x||_1 <= t}
//    s_r in {0}
MatrixProblem BuildLasso(
    const MatrixXd& A_lasso, const VectorXd& b_lasso, double lambda) {
  const int m = A_lasso.rows();
  const int n = A_lasso.cols();

  // Objective
  MatrixProblem problem;
  std::vector<Eigen::Triplet<double> > coeffs;
  AppendBlockTriplets(SparseIdentity(m), 0, 0, &coeffs);
  SparseXd Q(m+n+1, m+n+1);
  Q.setFromTriplets(coeffs.begin(), coeffs.end());
  VectorXd c = VectorXd::Zero(m+n+1);
  c(m+n) = lambda;

  // Constraints
  coeffs.clear();
  AppendBlockTriplets((-SparseIdentity(m+n+1)).eval(), 0, 0, &coeffs);
  AppendBlockTriplets(A_lasso, 0, m, &coeffs);
  SparseXd A(m+n+1, m+n+1);
  A.setFromTriplets(coeffs.begin(), coeffs.end());
  VectorXd b = VectorXd::Zero(m+n+1);
  b.head(m) = b_lasso;

  auto* constr = problem.add_epigraph_constraint();
  constr->set_function("Norm1");
  constr->set_x_index(m);
  constr->set_x_size(n);
  constr->set_t_index(m+n);

  GetSparseMatrixProto(A, problem.mutable_a());
  GetVectorProto(b, problem.mutable_b());
  GetVectorProto(c, problem.mutable_c());
  GetSparseMatrixProto(Q, problem.mutable_q());

  return problem;
}

// Construct lasso problem in epigraph form, cone version
//
// minimize_x 1/2*||Ax - b||^2 + lambda*||x||_1
// is equivalent to:
//
// minimize_r,t,x 1/2 r + lambda*t
// subject to
//   ||x||_1 <= t
//   ||1 - 2*b'Ax + b'b - r||   <=  1 + 2*b'Ax - b'b + r
//   ||    2Ax             ||_2
MatrixProblem BuildLassoCone(
    const MatrixXd& A_lasso, const VectorXd& b_lasso, double lambda) {
  const int m = A_lasso.rows();
  const int n = A_lasso.cols();
  MatrixProblem problem;

  // Objective
  VectorXd c = VectorXd::Zero(n+2);
  c(0) = 0.5;
  c(1) = lambda;

  // Constraints
  SparseXd A(m+n+3, n+2);
  VectorXd b = VectorXd::Zero(m+n+3);
  {
    int mi = 0;
    std::vector<Eigen::Triplet<double> > coeffs;

    // For ||x||_1 <= t constraint
    auto* constr = problem.add_epigraph_constraint();
    constr->set_function("Norm1");
    constr->set_t_index(mi+0);
    constr->set_x_index(mi+1);
    constr->set_x_size(n);
    AppendBlockTriplets(-MatrixXd::Identity(n+1, n+1), mi, 1, &coeffs);
    mi += n+1;

    // For cone constraint
    constr = problem.add_epigraph_constraint();
    constr->set_function("Norm2");
    constr->set_t_index(mi+0);
    constr->set_x_index(mi+1);
    constr->set_x_size(m+1);

    // 1 + 2*b'Ax - b'b + r
    coeffs.push_back(Eigen::Triplet<double>(mi, 0, -1));
    AppendBlockTriplets(-2*b_lasso.transpose()*A_lasso, mi, 2, &coeffs);
    b(mi) = 1 - b_lasso.dot(b_lasso);
    mi += 1;

    // 1 - 2*b'Ax + b'b - r
    coeffs.push_back(Eigen::Triplet<double>(mi, 0, 1));
    AppendBlockTriplets(2*b_lasso.transpose()*A_lasso, mi, 2, &coeffs);
    b(mi) = 1 + b_lasso.dot(b_lasso);
    mi += 1;

    // 2*Ax
    AppendBlockTriplets(-2*A_lasso, mi, 2, &coeffs);
    mi += m;

    CHECK_EQ(mi, A.rows());
    A.reserve(coeffs.size());
    A.setFromTriplets(coeffs.begin(), coeffs.end());
  }

  GetSparseMatrixProto(A, problem.mutable_a());
  GetVectorProto(b, problem.mutable_b());
  GetVectorProto(c, problem.mutable_c());
  return problem;
}

// Construct lasso problem in epigraph form, Ax = b version
//
// minimize_x 1/2*||Ax - b||^2 + lambda*||x||_1
// is equivalent to:
//
// minimize_r,x,t 1/2 r'r + lambda*t
// subject to
//   ||x||_1 <= t
//   Ax - b  = r
MatrixProblem BuildLasso2(
    const MatrixXd& A_lasso, const VectorXd& b_lasso, double lambda) {
  const int m = A_lasso.rows();
  const int n = A_lasso.cols();
  MatrixProblem problem;
  problem.set_n(m+n+1);

  // Objective
  VectorXd c = VectorXd::Zero(m+n+1);
  c(m+n) = lambda;
  SparseXd Q(m+n+1, m+n+1);
  {
    std::vector<Eigen::Triplet<double> > coeffs;
    AppendBlockTriplets(MatrixXd::Identity(m, m), 0, 0, &coeffs);
    Q.reserve(coeffs.size());
    Q.setFromTriplets(coeffs.begin(), coeffs.end());
  }

  // Linear constraints
  SparseXd A(m, m+n+1);
  VectorXd b = b_lasso;
  {
    // Ax - b = r
    std::vector<Eigen::Triplet<double> > coeffs;
    AppendBlockTriplets(-MatrixXd::Identity(m, m), 0, 0, &coeffs);
    AppendBlockTriplets(A_lasso, 0, m, &coeffs);
    A.reserve(coeffs.size());
    A.setFromTriplets(coeffs.begin(), coeffs.end());
  }

  // Cone constraints
  auto* constr = problem.add_epigraph_constraint();
  constr->set_function("Norm1");
  constr->set_x_index(m);
  constr->set_x_size(n);
  constr->set_t_index(m+n);

  GetSparseMatrixProto(A, problem.mutable_a());
  GetVectorProto(b, problem.mutable_b());
  GetVectorProto(c, problem.mutable_c());
  GetSparseMatrixProto(Q, problem.mutable_q());
  return problem;
}

// Construct a least squares problem in epigraph form
//
// minimize_x 1/2 ||Ax - b||^2
// is equivalent to:
//
// minimize_r,x 1/2 r'r
// subject to
//   Ax - b = r
MatrixProblem BuildLS(const MatrixXd& A_ls, const VectorXd& b_ls) {
  MatrixProblem problem;
  const int m = A_ls.rows();
  const int n = A_ls.cols();

  // Objective
  std::vector<Eigen::Triplet<double> > coeffs;
  AppendBlockTriplets(MatrixXd::Identity(m, m), 0, 0, &coeffs);
  SparseXd Q(m+n, m+n);
  Q.setFromTriplets(coeffs.begin(), coeffs.end());

  // Constraints
  coeffs.clear();
  AppendBlockTriplets(-MatrixXd::Identity(m, m), 0, 0, &coeffs);
  AppendBlockTriplets(A_ls, 0, m, &coeffs);
  SparseXd A(m, m+n);
  A.setFromTriplets(coeffs.begin(), coeffs.end());

  GetSparseMatrixProto(Q, problem.mutable_q());
  GetSparseMatrixProto(A, problem.mutable_a());
  GetVectorProto(b_ls, problem.mutable_b());

  return problem;
}

// Construct norm1 problem in epigraph form
//
// minimize_x ||x||_1
// is equivalent to:
//
// minimize_x,t t
// subject to
//  ||x||_t <= t
MatrixProblem BuildNorm1(int n) {
  MatrixProblem problem;

  // Objective
  VectorXd c = VectorXd::Zero(n+1);
  c(n) = 1;

  // Constraints
  std::vector<Eigen::Triplet<double> > coeffs;
  AppendBlockTriplets(-MatrixXd::Identity(n+1, n+1), 0, 0, &coeffs);
  SparseXd A(n+1, n+1);
  A.setFromTriplets(coeffs.begin(), coeffs.end());
  VectorXd b = VectorXd::Zero(n+1);

  GetSparseMatrixProto(A, problem.mutable_a());
  GetVectorProto(b, problem.mutable_b());
  GetVectorProto(c, problem.mutable_c());

  auto* constr = problem.add_epigraph_constraint();
  constr->set_function("Norm1");
  constr->set_x_index(0);
  constr->set_x_size(n);
  constr->set_t_index(n);

  return problem;
}

// Construct sparse inverse covariance estimation epigraph form
//
// minimize_X -log|X| + tr SX + lambda*||X||_1
// is equivalent to:
//
// minimize_X,t,w w + tr SX + lambda*t
// subject to
//   ||X||_1 <= t
//   -log|X| <= w
MatrixProblem BuildSparseInverseCovarianceEstimation(
    const MatrixXd& S, double lambda) {
  const int n = S.rows();
  CHECK_EQ(n, S.cols());

  MatrixProblem problem;
  problem.set_n(n*n+2);

  // Objective
  VectorXd c(n*n+2);
  c.segment(0, n*n) = Map<const VectorXd>(S.data(), n*n);
  c(n*n) = lambda;
  c(n*n+1) = 1;

  // Constraints
  SparseXd A(2*n*n+2, n*n+2);
  VectorXd b = VectorXd::Zero(2*n*n+2);
  {
    std::vector<Eigen::Triplet<double> > coeffs;
    int mi = 0;

    // ||X||_1 <= t
    AppendBlockTriplets((-SparseIdentity(n*n)).eval(), mi, 0, &coeffs);
    coeffs.push_back(Eigen::Triplet<double>(mi+n*n, n*n, -1));
    auto* constr = problem.add_epigraph_constraint();
    constr->set_function("Norm1");
    constr->set_x_index(mi);
    constr->set_x_size(n*n);
    constr->set_t_index(mi+n*n);
    mi += n*n+1;

    // -log|X| <= w
    AppendBlockTriplets((-SparseIdentity(n*n)).eval(), mi, 0, &coeffs);
    coeffs.push_back(Eigen::Triplet<double>(mi+n*n, n*n+1, -1));
    constr = problem.add_epigraph_constraint();
    constr->set_function("NegativeLogDet");
    constr->set_x_index(mi);
    constr->set_x_size(n*n);
    constr->set_t_index(mi+n*n);
    constr->mutable_params()->set_n(n);
    mi += n*n+1;

    CHECK_EQ(mi, A.rows());
    A.reserve(coeffs.size());
    A.setFromTriplets(coeffs.begin(), coeffs.end());
  }

  GetSparseMatrixProto(A, problem.mutable_a());
  GetVectorProto(b, problem.mutable_b());
  GetVectorProto(c, problem.mutable_c());

  return problem;
}

// Construct the Robust PCA in epigraph form
//
// minimize_L,S ||L||_* + lambda*||S||_1
// subject to
//   L + S = M
//
// is equivalent to
//
// minimize_L,w,t w + lambda*t
// subject to
//   ||L||_* <= w
//   ||M - L||_1 <= t
MatrixProblem BuildRobustPCA(const MatrixXd& M, double lambda) {
  const int m = M.rows();
  const int n = M.cols();

  MatrixProblem problem;
  problem.set_n(m*n+2);

  // Objective
  VectorXd c(m*n+2);
  c(m*n) = 1;
  c(m*n+1) = lambda;

  // Constraints
  SparseXd A(2*m*n+2, m*n+2);
  VectorXd b = VectorXd::Zero(2*m*n+2);
  {
    std::vector<Eigen::Triplet<double> > coeffs;
    int mi = 0;

    // ||L||_* <= w
    AppendBlockTriplets((-SparseIdentity(m*n+1)).eval(), mi, 0, &coeffs);
    auto* constr = problem.add_epigraph_constraint();
    constr->set_function("NormNuclear");
    constr->set_x_index(mi);
    constr->set_x_size(m*n);
    constr->set_t_index(mi+m*n);
    constr->mutable_params()->set_m(m);
    constr->mutable_params()->set_n(n);
    mi += m*n+1;

    // ||M - L||_1 <= t
    AppendBlockTriplets((SparseIdentity(m*n)).eval(), mi, 0, &coeffs);
    coeffs.push_back(Eigen::Triplet<double>(mi+m*n, m*n+1, -1));
    b.segment(mi, m*n) = Map<const VectorXd>(M.data(), m*n);
    constr = problem.add_epigraph_constraint();
    constr->set_function("Norm1");
    constr->set_x_index(mi);
    constr->set_x_size(m*n);
    constr->set_t_index(mi+m*n);
    mi += n*n+1;

    CHECK_EQ(mi, A.rows());
    A.reserve(coeffs.size());
    A.setFromTriplets(coeffs.begin(), coeffs.end());
  }

  GetSparseMatrixProto(A, problem.mutable_a());
  GetVectorProto(b, problem.mutable_b());
  GetVectorProto(c, problem.mutable_c());

  return problem;
}

MatrixProblem BuildLS_Cone_RowSplit(
  const MatrixXd& Ain, const VectorXd& bin, int N) {

  const int m = Ain.rows();
  const int n = Ain.cols();
  const int block_size = ceil(m / N);
  CHECK_EQ(bin.size(), m);

  MatrixProblem problem;
  SparseXd A(m+2*N+1, n+N+1);
  VectorXd b = VectorXd::Zero(m+2*N+1);
  VectorXd c = VectorXd::Zero(n+N+1);

  // Objective
  c(n+N) = 1;

  // Constraints
  {
    std::vector<Eigen::Triplet<double> > coeffs;

    int mi = 0;
    int i_index = 0;
    for (int i = 0; i < N; i++) {
      // ||Ai*x - bi|| <= ti
      int size_i = fmin(block_size, m - i_index);
      coeffs.push_back(Eigen::Triplet<double>(mi, n+i, -1));
      AppendBlockTriplets(-Ain.block(i_index, 0, size_i, n), mi+1, 0, &coeffs);
      b.segment(mi+1, size_i) = -bin.segment(i_index, size_i);

      problem.mutable_cone_constraints()->add_second_order_cone(size_i+1);
      mi += size_i+1;
      i_index += size_i;
    }

    coeffs.push_back(Eigen::Triplet<double>(mi, n+N, -1));
    mi += 1;
    for (int i = 0; i < N; i++) {
      coeffs.push_back(Eigen::Triplet<double>(mi, n+i, -1));
      mi += 1;
    }
    problem.mutable_cone_constraints()->add_second_order_cone(N+1);

    A.setFromTriplets(coeffs.begin(), coeffs.end());
  }

  GetSparseMatrixProto(A, problem.mutable_a());
  GetVectorProto(b, problem.mutable_b());
  GetVectorProto(c, problem.mutable_c());
  return problem;
}

MatrixProblem BuildLS_Cone(const MatrixXd& Ain, const VectorXd& bin) {
  const int m = Ain.rows();
  const int n = Ain.cols();
  CHECK_EQ(bin.size(), m);

  SparseXd A(m+1, n+1);
  VectorXd b = VectorXd::Zero(m+1);
  VectorXd c = VectorXd::Zero(n+1);
  std::vector<Eigen::Triplet<double> > coeffs;

  // Objective
  c(n) = 1;

  // Constraints, by row
  coeffs.push_back(Eigen::Triplet<double>(0, n, -1));

  AppendBlockTriplets(-Ain, 1, 0, &coeffs);
  b.segment(1, m) = -bin;

  A.setFromTriplets(coeffs.begin(), coeffs.end());

  MatrixProblem problem;
  GetSparseMatrixProto(A, problem.mutable_a());
  GetVectorProto(b, problem.mutable_b());
  GetVectorProto(c, problem.mutable_c());
  problem.mutable_cone_constraints()->add_second_order_cone(m+1);
  return problem;
}

MatrixProblem BuildProjLinearCone(
  const MatrixXd& Ain, const VectorXd& bin, const VectorXd& x0, double t0) {
  MatrixProblem problem;

  const int m = Ain.rows();
  const int n = Ain.cols();

  SparseXd A(m+1, n+1);
  VectorXd b(m+1);
  VectorXd c(n+1);

  // Set up objective
  c.head(n) = -x0;
  c[n] = -t0;
  SparseXd Q = SparseIdentity(n+1);

  // Set up constraints
  {
    int mi = 0;
    std::vector<Eigen::Triplet<double> > coeffs;
    coeffs.push_back(Eigen::Triplet<double>(0, n, -1));
    mi += 1;

    AppendBlockTriplets(-Ain, mi, 0, &coeffs);
    b.segment(mi, m) = -bin;
    mi += m;

    problem.mutable_cone_constraints()->add_second_order_cone(m+1);
    A.setFromTriplets(coeffs.begin(), coeffs.end());
  }

  GetSparseMatrixProto(A, problem.mutable_a());
  GetSparseMatrixProto(Q, problem.mutable_q());
  GetVectorProto(b, problem.mutable_b());
  GetVectorProto(c, problem.mutable_c());
  return problem;
}


// ||Ax - b||_2
Expression BuildLS_Cone(
    const Expression& A, const Expression& b, const std::string& var_id) {
  const int m = A.size().dim(0);
  const int n = A.size().dim(1);

  Expression output;
  Expression* norm2 = &output;
  norm2->set_expression_type(Expression::P_NORM);
  norm2->set_p(2);
  norm2->mutable_size()->add_dim(1);
  norm2->mutable_size()->add_dim(1);

  Expression* plus = norm2->add_arg();
  plus->set_expression_type(Expression::ADD);
  plus->mutable_size()->add_dim(m);
  plus->mutable_size()->add_dim(1);

  Expression* Ax = plus->add_arg();
  Ax->set_expression_type(Expression::MULTIPLY);
  Ax->mutable_size()->add_dim(m);
  Ax->mutable_size()->add_dim(1);
  Ax->add_arg()->CopyFrom(A);

  Expression* x = Ax->add_arg();
  x->set_expression_type(Expression::VARIABLE);
  x->mutable_variable()->set_variable_id(var_id);
  x->mutable_size()->add_dim(n);
  x->mutable_size()->add_dim(1);

  *plus->add_arg() = expression::Negate(b);
  return output;
}

// ||Ax - b||_2^2
Expression BuildLS(
    const Expression& A, const Expression& b, const std::string& var_id) {
  const int m = GetDimension(A, 0);
  const int n = GetDimension(A, 1);

  Expression output;
  output.set_expression_type(Expression::POWER);
  output.set_p(2);
  output.mutable_size()->add_dim(1);
  output.mutable_size()->add_dim(1);

  Expression* norm2 = output.add_arg();
  norm2->set_expression_type(Expression::P_NORM);
  norm2->set_p(2);
  norm2->mutable_size()->add_dim(1);
  norm2->mutable_size()->add_dim(1);

  Expression* plus = norm2->add_arg();
  plus->set_expression_type(Expression::ADD);
  plus->mutable_size()->add_dim(m);
  plus->mutable_size()->add_dim(1);

  Expression* Ax = plus->add_arg();
  Ax->set_expression_type(Expression::MULTIPLY);
  Ax->mutable_size()->add_dim(m);
  Ax->mutable_size()->add_dim(1);
  Ax->add_arg()->CopyFrom(A);

  Expression* x = Ax->add_arg();
  x->set_expression_type(Expression::VARIABLE);
  x->mutable_variable()->set_variable_id(var_id);
  x->mutable_size()->add_dim(n);
  x->mutable_size()->add_dim(1);

  *plus->add_arg() = expression::Negate(b);
  return output;
}

// 0.5 ||Ax - b||_2^2 - lambda*||x||_1
Expression BuildLasso(
    const Expression& A, const Expression& b, double lambda,
    const std::string& var_id) {
  const int n = GetDimension(A, 1);

  Expression sum_squares = BuildLS(A, b, var_id);
  Expression norm1 = BuildNorm1(n, var_id);

  Expression lambda_norm1;
  lambda_norm1.set_expression_type(Expression::MULTIPLY);
  *lambda_norm1.mutable_size() = kScalarSize;
  *lambda_norm1.add_arg() = expression::ScalarConstant(lambda);
  *lambda_norm1.add_arg() = norm1;

  Expression plus;
  plus.set_expression_type(Expression::ADD);
  *plus.mutable_size() = kScalarSize;
  *plus.add_arg() = sum_squares;
  *plus.add_arg() = lambda_norm1;

  return plus;
}

// ||Ax - b||_1
Expression BuildMinAbsError(
    const Expression& A, const Expression& b, const std::string& var_id) {
  Expression output;
  output.set_expression_type(Expression::P_NORM);
  output.set_p(1);
  output.mutable_size()->add_dim(1);
  output.mutable_size()->add_dim(1);

  Expression* plus = output.add_arg();
  plus->set_expression_type(Expression::ADD);
  plus->mutable_size()->add_dim(10);
  plus->mutable_size()->add_dim(1);

  Expression* Ax = plus->add_arg();
  Ax->set_expression_type(Expression::MULTIPLY);
  Ax->mutable_size()->add_dim(10);
  Ax->mutable_size()->add_dim(1);
  Ax->add_arg()->CopyFrom(A);

  Expression* x = Ax->add_arg();
  x->set_expression_type(Expression::VARIABLE);
  x->mutable_variable()->set_variable_id(var_id);
  x->mutable_size()->add_dim(5);
  x->mutable_size()->add_dim(1);

  *plus->add_arg() = expression::Negate(b);
  return output;
}

// 0.1*||x||_1
Expression BuildNorm1(int n, const std::string& var_id) {
  Expression output;
  output.set_expression_type(Expression::P_NORM);
  output.set_p(1);
  output.mutable_size()->add_dim(1);
  output.mutable_size()->add_dim(1);

  Expression* arg = output.add_arg();
  arg->set_expression_type(Expression::VARIABLE);
  arg->mutable_size()->add_dim(n);
  arg->mutable_size()->add_dim(1);
  arg->mutable_variable()->set_variable_id(var_id);
  return output;
}
