#include "epsilon/expression/expression.h"

#include "epsilon/util/logging.h"

namespace expression {

// Helper functions
Expression ScalarConstant(double scalar) {
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

Expression Variable(int m, int n, const std::string& var_id, int offset) {
  Expression t;
  t.set_expression_type(Expression::VARIABLE);
  t.mutable_size()->add_dim(m);
  t.mutable_size()->add_dim(n);
  t.mutable_variable()->set_variable_id(var_id);
  t.mutable_variable()->set_offset(offset);
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


Expression PNorm(const Expression& x, int p) {
  Expression expr;
  expr.set_expression_type(Expression::P_NORM);
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

Expression Norm2Elementwise(const std::vector<Expression>& args) {
  CHECK_GT(args.size(), 0);
  const int m = GetDimension(args[0], 0);

  Expression expr;
  expr.set_expression_type(Expression::NORM_2_ELEMENTWISE);
  *expr.mutable_size() = CreateSize(m, 1);

  for (const Expression& arg : args) {
    CHECK_EQ(m, GetDimension(arg, 0));
    *expr.add_arg() = arg;
  }

  return expr;
}

}  // namespace expression

std::string ExpressionKey(const Expression& expression) {
  return expression.SerializeAsString();
}

int GetDimension(const Expression& expr) {
  CHECK_EQ(expr.size().dim_size(), 2) << expr.DebugString();
  return expr.size().dim(0)*expr.size().dim(1);
}

int GetDimension(const Expression& expr, int dim) {
  CHECK_EQ(expr.size().dim_size(), 2);
  return expr.size().dim(dim);
}

const Expression& GetOnlyArg(const Expression& expr) {
  CHECK_EQ(expr.arg_size(), 1);
  return expr.arg(0);
}

const Expression& GetLeaf(const Expression& expr) {
  const Expression* e = &expr;
  for (; e->arg_size() > 0; e = &e->arg(0));
  return *e;
}

int GetVariableDimension(const Expression& expr) {
  std::set<const Expression*, VariableIdCompare> retval;
  GetVariables(expr, &retval);
  int n = 0;
  for (auto iter : retval) {
    n += GetDimension(*iter);
  }
  return n;
}

void GetDataLocations(const Expression& expression,
                      std::unordered_set<std::string>* locations) {
  if (expression.constant().data_location() != "")
    locations->insert(expression.constant().data_location());
  for (const Expression& arg : expression.arg()) {
    GetDataLocations(arg, locations);
  }
}

void GetConstant(const Expression& expression, Constant* constant) {
  bool transpose = false;
  const Expression* e = &expression;
  while (e->arg_size() > 0) {
    CHECK_EQ(1, e->arg_size());
    CHECK_EQ(Expression::TRANSPOSE, e->expression_type());
    e = &e->arg(0);
    transpose = !transpose;
  }
  CHECK_EQ(Expression::CONSTANT, e->expression_type());
  CHECK(e->has_constant());
  constant->CopyFrom(e->constant());
  constant->set_transpose(transpose);
}

Size CreateSize(int m, int n) {
  Size size;
  size.add_dim(m);
  size.add_dim(n);
  return size;
}

Expression RemoveArgs(const Expression& input) {
  Expression output = input;
  output.clear_arg();
  return output;
}

bool VariableIdCompare::operator() (const Expression* a, const Expression* b) {
  CHECK_EQ(a->expression_type(), Expression::VARIABLE);
  CHECK_EQ(b->expression_type(), Expression::VARIABLE);
  return a->variable().variable_id() < b->variable().variable_id();
}

void GetVariables (
    const Expression& expr,
    std::set<const Expression*, VariableIdCompare>* vars) {
  if (expr.expression_type() == Expression::VARIABLE) {
    vars->insert(&expr);
  }
  for (const Expression& arg : expr.arg()) {
    GetVariables(arg, vars);
  }
}

bool IsScalarConstant(const Expression& expr) {
  if (expr.arg_size() == 0) {
    return (expr.expression_type() == Expression::CONSTANT &&
            GetDimension(expr) == 1);
  }

  for (const Expression& arg : expr.arg()) {
    if (!IsScalarConstant(arg))
      return false;
  }
  return true;
}

double GetScalarConstant(const Expression& expr) {
  if (expr.arg_size() == 0) {
    CHECK(expr.expression_type() == Expression::CONSTANT);
    return expr.constant().scalar();
  }

  if (expr.expression_type() == Expression::ADD) {
    double retval = 0;
    for (const Expression& arg : expr.arg()) {
      retval += GetScalarConstant(arg);
    }
  } else if (expr.expression_type() == Expression::MULTIPLY) {
    double retval = 1;
    for (const Expression& arg : expr.arg()) {
      retval *= GetScalarConstant(arg);
    }
  }

  LOG(FATAL) << "Unsupported scalar operator: " << expr.expression_type();
  return 0;
}

void AddVariableOffsets(
    Expression* expression,
    VariableOffsetMap* var_offsets) {
  if (expression->expression_type() == Expression::VARIABLE) {
    expression->mutable_variable()->set_offset(var_offsets->Get(*expression));
  }
  for (Expression& arg : *expression->mutable_arg()) {
    AddVariableOffsets(&arg, var_offsets);
  }
}
