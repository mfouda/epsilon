#ifndef EXPRESSION_EXPRESSION_H
#define EXPRESSION_EXPRESSION_H

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "epsilon/expression.pb.h"

namespace expression {

Expression Add(const Expression& a, const Expression& b);
Expression Add(const Expression& a, const Expression& b, const Expression& c);
Expression Add(const std::vector<Expression>& args);
Expression HStack(const std::vector<Expression>& args);
Expression VStack(const std::vector<Expression>& args);
Expression Index(int i, int m, const Expression& input);
Expression Index(int i, int m, int j, int n, const Expression& input);
Expression LogDet(const Expression &x);
Expression Multiply(const Expression& left, const Expression& right);
Expression MultiplyElementwise(const Expression& left, const Expression& right);
Expression Negate(const Expression& x);
Expression Norm2Elementwise(const std::vector<Expression>& args);
Expression PNorm(const Expression& x, int p);
Expression Power(const Expression& x, int p);
Expression ScalarConstant(double scalar);
Expression Sum(const Expression& x);
Expression Variable(int m, int n, const std::string& var_id, int offset = 0);

}  // namespace expression

std::string ExpressionKey(const Expression& expression);
int GetDimension(const Expression& expression);
int GetDimension(const Expression& expression, int dim);
int GetVariableDimension(const Expression& expr);

const Expression& GetOnlyArg(const Expression& expression);
const Expression& GetLeaf(const Expression& expression);
void GetDataLocations(const Expression& expression,
                      std::unordered_set<std::string>* locations);

Size CreateSize(int m, int n);
const Size kScalarSize = CreateSize(1,1);

Expression RemoveArgs(const Expression& expression);

struct VariableIdCompare {
  bool operator() (const Expression* a, const Expression* b);
};

class VariableOffsetMap {
 public:
  VariableOffsetMap() : n_(0) {}

  // Get the variable offset for expr (must be VARIABLE type) inserting it into
  // the map if it does not already exist.
  int Get(const Expression& expr);
  int n() const { return n_; }

 private:
  std::unordered_map<std::string, int> offsets_;
  int n_;
};

void AddVariableOffsets(
    Expression* expression,
    VariableOffsetMap* var_offsets);

void GetVariables (
    const Expression& expr,
    std::set<const Expression*, VariableIdCompare>* vars);

// Dealing with scalars
bool IsScalarConstant(const Expression& expr);
double GetScalarConstant(const Expression& expr);

#endif  // EXPRESSION_EXPRESSION_H
