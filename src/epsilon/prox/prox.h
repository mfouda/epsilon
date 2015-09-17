#ifndef EPSILON_OPERATORS_PROX_H
#define EPSILON_OPERATORS_PROX_H

#include <vector>
#include <memory>

#include "epsilon/expression.pb.h"
#include "epsilon/expression/var_offset_map.h"
#include "epsilon/vector/vector_operator.h"

std::unique_ptr<VectorOperator> CreateProxOperator(
    double lambda,
    const Expression& f_expr,
    const VariableOffsetMap& var_map);

// Arguments to the proximal operator, lambda*f(A*x + b)
class ProxOperatorArg {
 public:
  ProxOperatorArg(
      double lambda,
      const Expression* f_expr,
      const VariableOffsetMap* var_map)
      : lambda_(lambda), f_expr_(f_expr), var_map_(var_map) {}

  double lambda() const { return lambda_; };

  // Ax+b in expression form
  const Expression& f_expr() const { return *f_expr_; }
  const VariableOffsetMap& var_map() const { return *var_map_; }

 private:
  double lambda_;

  // Not owned by us
  const Expression* f_expr_;
  const VariableOffsetMap* var_map_;
};

// Abstract interface for proximal operator implementations
class ProxOperator {
 public:
  virtual void Init(const ProxOperatorArg& arg) {}
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& v) = 0;
};

template<class T>
bool RegisterProxOperator(const std::string& id);
#define REGISTER_PROX_OPERATOR(T) bool registered_##T = RegisterProxOperator<T>(#T)



#endif  // EPSILON_OPERATORS_PROX_H
