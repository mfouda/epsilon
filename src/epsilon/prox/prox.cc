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

// TODO(mwytock): Utility functions that should likely go elsewhere

// Generic VectorOperator which handles the generic preprocessing and delegates
// work to ProxOperator.
class ProxVectorOperator : public VectorOperator {
 public:
  ProxVectorOperator(
      double lambda,
      const Expression& f_expr,
      const VariableOffsetMap& var_map)
      : lambda_(lambda), f_expr_(f_expr), var_map_(var_map) {}

  void Init() override {
    Preprocess();
    g_prox_->Init(ProxOperatorArg(alpha_*lambda_, nullptr, g_expr_, &var_map_));
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    return g_prox_->Apply(v - lambda_*c_);
  }

 private:
  void Preprocess();

  // Original function parameters
  double lambda_;
  Expression f_expr_;
  const VariableOffsetMap& var_map_;

  // Preprocessed parameters, f(x) = alpha*g(x) + c'x
  const Expression* g_expr_;
  double alpha_;
  Eigen::VectorXd c_;
  std::unique_ptr<ProxOperator> g_prox_;
};

std::unordered_map<
  std::string,
  std::function<std::unique_ptr<ProxOperator>()>>* kProxOperatorMap;

std::unordered_map<
  std::string,
  std::function<std::unique_ptr<BlockProxOperator>()>>* kBlockProxOperatorMap;

// Preprocess f(x) to extract f(x) = alpha*g(Ax + b) + c'x.
// NOTE(mwytock): We assume the input expression has already undergone
// processing and thus we dont need to handle fully general expressions here.
void ProxVectorOperator::Preprocess() {
  const int n  = var_map_.n();

  // Add affine term
  g_expr_ = &f_expr_;
  if (g_expr_->expression_type() == Expression::ADD) {
    CHECK_EQ(2, f_expr_.arg_size());
    g_expr_ = &f_expr_.arg(0);

    DynamicMatrix A = DynamicMatrix::Zero(1, n);
    DynamicMatrix b = DynamicMatrix::Zero(1, 1);
    BuildAffineOperator(f_expr_.arg(1), var_map_, &A, &b);
    c_ = ToVector(A.AsDense());
  } else {
    c_ = Eigen::VectorXd::Zero(n);
  }

  // Multiply scalar constant
  if (g_expr_->expression_type() == Expression::MULTIPLY &&
      IsScalarConstant(g_expr_->arg(0))) {
    alpha_ = GetScalarConstant(g_expr_->arg(0));
    g_expr_ = &g_expr_->arg(1);
  } else {
    alpha_ = 1;
  }

  // Get prox function and arg
  CHECK(kProxOperatorMap != nullptr) << "No registered operators";
  auto iter = kProxOperatorMap->find(g_expr_->proximal_operator().name());
  if (iter == kProxOperatorMap->end()) {
    LOG(FATAL) << "No proximal operator for "
               << g_expr_->proximal_operator().name() << "\n"
               << g_expr_->DebugString();
  }
  g_prox_ = iter->second();

  VLOG(2) << "Preprocess, alpha = " << alpha_
          << ", c = " << VectorDebugString(c_);
}

std::unique_ptr<VectorOperator> CreateProxOperator(
    double lambda,
    const Expression& f_expr,
    const VariableOffsetMap& var_map) {
  return std::unique_ptr<VectorOperator>(new ProxVectorOperator(
      lambda, f_expr, var_map));
}

class ProxBlockVectorOperator final : public BlockVectorOperator {
 public:
  ProxBlockVectorOperator(
      double lambda,
      BlockMatrix A,
      const Expression& f_expr)
      : prox_vector_operator_(lambda, f_expr, var_map_),
      lambda_(lambda),
      A_(A),
      f_expr_(f_expr) {
    var_map_.Insert(f_expr);
  }

  void Init() override {
    CHECK(kBlockProxOperatorMap != nullptr) << "No registered operators";
    auto iter = kBlockProxOperatorMap->find(f_expr_.proximal_operator().name());
    if (iter == kBlockProxOperatorMap->end()) {
      block_vector_prox_ = false;
      prox_vector_operator_.Init();
    } else {
      block_vector_prox_ = true;
      prox_ = iter->second();
      prox_->Init(ProxOperatorArg(lambda_, &A_, &f_expr_, &var_map_));
    }
  }

  // Currently we just map to ProxVectorOperator. In future we will replace that
  // functionality here and avoid unnecessary copying.
  virtual BlockVector Apply(const BlockVector& v) override {
    if (block_vector_prox_)
      return prox_->Apply(v);

    Eigen::VectorXd v_vec = Eigen::VectorXd::Zero(var_map_.n());
    for (auto iter : var_map_.offsets()) {
      if (v.has_key(iter.first))
        v_vec.segment(iter.second, var_map_.Size(iter.first)) = v(iter.first);
    }
    Eigen::VectorXd x_vec = prox_vector_operator_.Apply(v_vec);
    BlockVector x;
    for (auto iter : var_map_.offsets()) {
      x(iter.first) = x_vec.segment(iter.second, var_map_.Size(iter.first));
    }
    return x;
  }

 private:
  VariableOffsetMap var_map_;
  ProxVectorOperator prox_vector_operator_;

  bool block_vector_prox_;
  double lambda_;
  BlockMatrix A_;
  Expression f_expr_;
  std::unique_ptr<BlockProxOperator> prox_;
};

std::unique_ptr<BlockVectorOperator> CreateProxOperator(
    double lambda,
    BlockMatrix A,
    const Expression& f_expr) {
  return std::unique_ptr<BlockVectorOperator>(new ProxBlockVectorOperator(
      lambda, A, f_expr));
}
