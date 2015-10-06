#include "epsilon/expression/expression_util.h"

#include <glog/logging.h>

bool VariableIdCompare::operator() (const Expression* a, const Expression* b) {
  CHECK_EQ(a->expression_type(), Expression::VARIABLE);
  CHECK_EQ(b->expression_type(), Expression::VARIABLE);
  return a->variable().variable_id() < b->variable().variable_id();
}

VariableSet GetVariables(const Problem& problem) {
  VariableSet vars = GetVariables(problem.objective());
  for (const Expression& constr : problem.constraint()) {
    VariableSet sub_vars = GetVariables(constr);
    vars.insert(sub_vars.begin(), sub_vars.end());
  }
  return vars;
}

VariableSet GetVariables(const Expression& expr) {
  VariableSet vars;
  if (expr.expression_type() == Expression::VARIABLE) {
    vars.insert(&expr);
  }

  for (const Expression& arg : expr.arg()) {
    VariableSet sub_vars = GetVariables(arg);
    vars.insert(sub_vars.begin(), sub_vars.end());
  }
  return vars;
}

int GetVariableDimension(const VariableSet& var_set) {
  int n = 0;
  for (const Expression* expr : var_set) {
    n += GetDimension(*expr);
  }
  return n;
}

uint64_t VariableParameterId(
    uint64_t problem_id, const std::string& variable_id_str) {
  return problem_id ^ std::hash<std::string>()(variable_id_str);
}

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

const Expression& GetRightmostLeaf(const Expression& expr) {
  const Expression* e = &expr;
  for (; e->arg_size() > 0; e = &e->arg(e->arg_size()-1));
  return *e;
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
