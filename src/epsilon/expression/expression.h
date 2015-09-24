#ifndef EPSILON_EXPRESSION_EXPRESSION_H
#define EPSILON_EXPRESSION_EXPRESSION_H

#include <string>
#include <vector>

#include "epsilon/expression.pb.h"

namespace expression {

Expression Add(const Expression& a, const Expression& b);
Expression Add(const Expression& a, const Expression& b, const Expression& c);
Expression Add(const std::vector<Expression>& args);
Expression Constant(double scalar);
Expression HStack(const std::vector<Expression>& args);
Expression Index(int i, int m, const Expression& input);
Expression Index(int i, int m, int j, int n, const Expression& input);
Expression Indicator(const Cone::Type& cone_type, const Expression& arg);
Expression LogDet(const Expression &x);
Expression Multiply(const Expression& left, const Expression& right);
Expression MultiplyElementwise(const Expression& left, const Expression& right);
Expression Negate(const Expression& x);
Expression NormP(const Expression& x, int p);
Expression NormPQ(const Expression& x, int p, int q);
Expression Power(const Expression& x, int p);
Expression Reshape(const Expression& x, int m, int n);
Expression Sum(const Expression& x);
Expression VStack(const std::vector<Expression>& args);
Expression Variable(int m, int n, const std::string& var_id);

}  // namespace expression

#endif  // EPSILON_EXPRESSION_EXPRESSION_H
