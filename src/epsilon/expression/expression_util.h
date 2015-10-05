#ifndef EPSILON_EXPRESSION_EXPRESSION_UTIL_H
#define EPSILON_EXPRESSION_EXPRESSION_UTIL_H

#include <string>
#include <set>

#include "epsilon/expression.pb.h"

struct VariableIdCompare {
  bool operator() (const Expression* a, const Expression* b);
};
typedef std::set<const Expression*, VariableIdCompare> VariableSet;

VariableSet GetVariables(const Problem& problem);
VariableSet GetVariables(const Expression& expr);
int GetVariableDimension(const VariableSet& expr);

int GetDimension(const Expression& expression);
int GetDimension(const Expression& expression, int dim);

const Expression& GetOnlyArg(const Expression& expression);
const Expression& GetLeaf(const Expression& expression);
const Expression& GetRightmostLeaf(const Expression& expression);

Size CreateSize(int m, int n);
const Size kScalarSize = CreateSize(1,1);

Expression RemoveArgs(const Expression& expression);

// Dealing with scalars
bool IsScalarConstant(const Expression& expr);
double GetScalarConstant(const Expression& expr);

uint64_t VariableParameterId(uint64_t problem_id, const std::string& variable_id_str);

#endif  // EPSILON_EXPRESSION_EXPRESSION_UTIL_H
