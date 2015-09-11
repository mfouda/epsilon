#include "epsilon/expression/expression.h"

#include <glog/logging.h>

#include "epsilon/expression/expression_util.h"

namespace expression {

// Helper functions
Expression Constant(double scalar) {
  Expression t;
  t.set_expression_type(Expression::CONSTANT);
  t.mutable_size()->add_dim(1);
  t.mutable_size()->add_dim(1);
  t.mutable_constant()->set_scalar(scalar);
  return t;
}

Expression Negate(const Expression& x) {
  Expression t;
  t.set_expression_type(Expression::NEGATE);
  t.mutable_size()->CopyFrom(x.size());
  t.add_arg()->CopyFrom(x);
  return t;
}

Expression Sum(const Expression& x) {
  Expression t;
  t.set_expression_type(Expression::SUM);
  *t.mutable_size() = kScalarSize;
  t.add_arg()->CopyFrom(x);
  return t;
}

Expression Variable(int m, int n, const std::string& var_id) {
  Expression t;
  t.set_expression_type(Expression::VARIABLE);
  t.mutable_size()->add_dim(m);
  t.mutable_size()->add_dim(n);
  t.mutable_variable()->set_variable_id(var_id);
  return t;
}

Expression Index(int i, int m, int j, int n, const Expression& input) {
  Expression expr;
  expr.set_expression_type(Expression::INDEX);
  *expr.mutable_size() = CreateSize(m, n);
  *expr.add_arg() = input;

  {
    // First slice
    Slice* slice = expr.add_key();
    slice->set_start(i);
    slice->set_stop(i+m);
    slice->set_step(1);
  }

  {
    // Second slice
    Slice *slice = expr.add_key();
    slice->set_start(j);
    slice->set_stop(j+n);
    slice->set_step(1);
  }

  return expr;
}

Expression Index(int i, int m, const Expression& input) {
  return Index(i, m, 0, GetDimension(input, 1), input);
}

// A*B
Expression Multiply(const Expression& A, const Expression& B) {
  CHECK_EQ(GetDimension(A, 1), GetDimension(B, 0));

  Expression expr;
  expr.set_expression_type(Expression::MULTIPLY);
  *expr.mutable_size() = CreateSize(GetDimension(A, 0), GetDimension(B, 1));
  *expr.add_arg() = A;
  *expr.add_arg() = B;

  return expr;
}

// A.*B
Expression MultiplyElementwise(const Expression& A, const Expression& B) {
  CHECK_EQ(GetDimension(A, 0), GetDimension(B, 0));
  CHECK_EQ(GetDimension(A, 1), GetDimension(B, 1));

  Expression expr;
  expr.set_expression_type(Expression::MULTIPLY_ELEMENTWISE);
  *expr.mutable_size() = A.size();
  *expr.add_arg() = A;
  *expr.add_arg() = B;

  return expr;
}

Expression Add(const Expression& a, const Expression& b, const Expression& c) {
  return Add({a, b, c});
}

Expression Add(const Expression& a, const Expression& b) {
  return Add({a, b});
}

// A + B + ...
Expression Add(const std::vector<Expression>& args) {
  CHECK_GE(args.size(), 1);
  Expression expr;
  expr.set_expression_type(Expression::ADD);
  *expr.mutable_size() = args[0].size();

  for (const Expression& arg : args) {
    *expr.add_arg() = arg;
  }
  return expr;
}

// hstack(A, B, ...)
Expression HStack(const std::vector<Expression>& args) {
  CHECK_GE(args.size(), 1);
  Expression expr;
  expr.set_expression_type(Expression::HSTACK);

  int rows = GetDimension(args[0], 0);
  int cols = 0;
  for (const Expression& arg : args) {
    CHECK_EQ(GetDimension(arg, 0), rows);
    cols += GetDimension(arg, 1);
    *expr.add_arg() = arg;
  }

  expr.mutable_size()->add_dim(rows);
  expr.mutable_size()->add_dim(cols);
  return expr;
}

// vstack(A, B, ...)
Expression VStack(const std::vector<Expression>& args) {
  CHECK_GE(args.size(), 1);
  Expression expr;
  expr.set_expression_type(Expression::VSTACK);

  int rows = 0;
  int cols = GetDimension(args[0], 1);
  for (const Expression& arg : args) {
    CHECK_EQ(GetDimension(arg, 1), cols);
    rows += GetDimension(arg, 0);
    *expr.add_arg() = arg;
  }

  expr.mutable_size()->add_dim(rows);
  expr.mutable_size()->add_dim(cols);
  return expr;
}


Expression NormP(const Expression& x, int p) {
  Expression expr;
  expr.set_expression_type(Expression::NORM_P);
  expr.set_p(p);
  *expr.mutable_size() = kScalarSize;
  *expr.add_arg() = x;
  return expr;
}

Expression Power(const Expression& x, int p) {
  Expression expr;
  expr.set_expression_type(Expression::POWER);
  expr.set_p(p);
  *expr.mutable_size() = kScalarSize;
  *expr.add_arg() = x;
  return expr;
}

Expression LogDet(const Expression& x) {
  Expression expr;
  expr.set_expression_type(Expression::LOG_DET);
  *expr.mutable_size() = kScalarSize;
  *expr.add_arg() = x;
  return expr;
}

Expression NormPQ(const Expression& x, int p, int q) {
  Expression expr;
  expr.set_expression_type(Expression::NORM_PQ);
  *expr.mutable_size() = kScalarSize;
  expr.set_p(p);
  expr.set_q(q);
  *expr.add_arg() = x;
  return expr;
}

Expression Indicator(const Cone::Type& cone_type, const Expression& arg) {
  Expression expr;
  expr.set_expression_type(Expression::INDICATOR);
  *expr.mutable_size() = kScalarSize;
  expr.mutable_cone()->set_cone_type(cone_type);
  *expr.add_arg() = arg;
  return expr;
}

}  // namespace expression
