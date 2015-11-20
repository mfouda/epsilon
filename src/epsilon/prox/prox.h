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
      const AffineOperator& affine_arg,
      const AffineOperator& affine_constraint)
      : affine_arg_(affine_arg),
        affine_constraint_(affine_constraint) {}

  const AffineOperator& affine_arg() const { return affine_arg_; }
  const AffineOperator& affine_constraint() const { return affine_constraint_; }

 private:
  // Not owned by us
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

#define REGISTER_PROX_OPERATOR(type, T) bool registered_##T = RegisterProxOperator<T>(type)

#endif  // EPSILON_OPERATORS_PROX_H
