// A general interface to a proximal operator for a function, f : R^n -> R, of
// the form:
// lambda*f(x) = lambda*(alpha*g(Ax + b) + c'x)
//
// Many operators require special structure in A, e.g. diagonal or identity.
//
// TODO(mwytock): All proximal operators can handle an orthogonal transform,
// that shoul be baked in.
// TODO(mwytock): Figure out more precisely the interface for matrix
// variables.
// TODO(mwytock): Handle vector elementwise functions more automatically?
// TODO(mwytock): Handle b term more automatically?

#include "epsilon/prox/prox.h"

#include <glog/logging.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

#include "epsilon/expression/expression_util.h"
#include "epsilon/affine/affine.h"
#include "epsilon/vector/vector_util.h"

std::unordered_map<
  ProxFunction::Type,
  std::function<std::unique_ptr<ProxOperator>()>>* kProxOperatorMap;

std::unique_ptr<ProxOperator> CreateProxOperator(ProxFunction::Type type) {
  CHECK(kProxOperatorMap != nullptr) << "No registered operators";
  auto iter = kProxOperatorMap->find(type);
  if (iter == kProxOperatorMap->end()) {
    LOG(FATAL) << "No proximal operator for "
               << ProxFunction::Type_Name(type);
  }
  return iter->second();
}
