#ifndef EPSILON_OPERATORS_PROX_H
#define EPSILON_OPERATORS_PROX_H

#include <vector>
#include <memory>

#include "epsilon/affine/affine.h"
#include "epsilon/expression.pb.h"
#include "epsilon/expression/var_offset_map.h"

class ProxOperatorArg {
 public:
  ProxOperatorArg(
      const ProxFunction& prox_function,
      const AffineOperator& affine_arg,
      const AffineOperator& affine_constraint)
      : prox_function_(prox_function),
        affine_arg_(affine_arg),
        affine_constraint_(affine_constraint) {}

  const ProxFunction& prox_function() const { return prox_function_; }
  const AffineOperator& affine_arg() const { return affine_arg_; }
  const AffineOperator& affine_constraint() const { return affine_constraint_; }

 private:
  // Not owned by us
  const ProxFunction& prox_function_;
  const AffineOperator& affine_arg_;
  const AffineOperator& affine_constraint_;
};

// Abstract interface for proximal operator implementations
class ProxOperator {
 public:
  virtual void Init(const ProxOperatorArg& arg) {}
  virtual BlockVector Apply(const BlockVector& v) = 0;
};

// Create a generalized proximal operator for expression
// argmin_x f(H(x)) + (1/2)||A(x) - v||^2
std::unique_ptr<ProxOperator> CreateProxOperator(ProxFunction::Type type);

extern std::unordered_map<
  ProxFunction::Type,
  std::function<std::unique_ptr<ProxOperator>()>>* kProxOperatorMap;

template<class T>
bool RegisterProxOperator(ProxFunction::Type type) {
  if (kProxOperatorMap == nullptr) {
    kProxOperatorMap = new std::unordered_map<
      ProxFunction::Type,
      std::function<std::unique_ptr<ProxOperator>()>>();
  }

  kProxOperatorMap->insert(std::make_pair(
      type, [] {
        return std::unique_ptr<T>(new T);
      }));
  return true;
}

// NOTE(mwytock) C preprocessor nastiness
#define REGISTER_VAR(type, T) registered_ ## type ## _ ##T
#define REGISTER_PROX_OPERATOR(type, T) \
  bool REGISTER_VAR(type, T) = RegisterProxOperator<T>(ProxFunction::type)

#endif  // EPSILON_OPERATORS_PROX_H
