
#include "distopt/expression/expression_testutil.h"

#include <grpc++/client_context.h>
#include <gflags/gflags.h>

#include "distopt/data.pb.h"
#include "distopt/util/string.h"
#include "distopt/file/file.h"

DEFINE_string(test_data_prefix, "/mem/test",
	      "prefix for storing test data");

Expression TestConstant(const MatrixXd& A) {
  std::string loc = StringPrintf(
      "%s/random/%d", FLAGS_test_data_prefix.c_str(), rand());

  Expression expr;
  expr.set_expression_type(Expression::CONSTANT);
  expr.mutable_constant()->set_data_location(loc);
  expr.mutable_size()->add_dim(A.rows());
  expr.mutable_size()->add_dim(A.cols());

  WriteMatrixData(A, loc);
  return expr;
}

Expression TestVariable(int m, int  n) {
  Expression expr;
  expr.set_expression_type(Expression::VARIABLE);
  expr.mutable_variable()->set_variable_id(StringPrintf("var:%d", rand()));
  expr.mutable_size()->add_dim(m);
  expr.mutable_size()->add_dim(n);
  return expr;
}

MatrixXd RandomConstant(int m, int n, Expression* expr) {
  MatrixXd A = MatrixXd::Random(m, n);
  Expression expr2 = TestConstant(A);
  if (expr != nullptr) {
    *expr = expr2;
  }
  return A;
}

Expression RandomConstantOp(
  int m, int n, const std::string& input_key, MatrixXd* A) {
  Expression c_expr;

  if (A != nullptr)
    *A = RandomConstant(m, n, &c_expr);
  else
    RandomConstant(m, n, &c_expr);

  Expression expr;
  expr.set_expression_type(Expression::MULTIPLY);
  expr.mutable_size()->add_dim(m);
  expr.mutable_size()->add_dim(1);
  expr.mutable_constant()->CopyFrom(c_expr.constant());

  Expression* var_expr = expr.add_arg();
  var_expr->set_expression_type(Expression::VARIABLE);
  var_expr->mutable_variable()->set_variable_id(input_key);
  var_expr->mutable_size()->add_dim(n);
  var_expr->mutable_size()->add_dim(1);

  return expr;
}
