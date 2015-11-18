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

class ProxBlockVectorOperator final : public BlockVectorOperator {
 public:
  ProxBlockVectorOperator(
      double lambda,
      BlockMatrix A,
      const Expression& f_expr)
      : lambda_(lambda),
      A_(A),
      f_expr_(f_expr) {
    CHECK(kProxOperatorMap != nullptr) << "No registered operators";
  }

  void Init() override {
    AT_ = A_.Transpose();

    Preprocess();
    InitScaling();
    CHECK_EQ(Expression::PROX_FUNCTION, g_expr_->expression_type());
    ProxFunction::Type type = g_expr_->prox_function().prox_function_type();
    auto iter = kProxOperatorMap->find(type);
    if (iter == kProxOperatorMap->end()) {
        LOG(FATAL) << "No proximal operator for "
                   << ProxFunction::Type_Name(type);
    }

    prox_ = iter->second();
    prox_->Init(ProxOperatorArg(alpha_*lambda_, &A_, g_expr_, nullptr));
  }

  virtual BlockVector Apply(const BlockVector& v) override {
    BlockVector v_c = AT_*v - lambda_*c_;
    return prox_->Apply(v_c);
  }

 private:
  void Preprocess();
  void InitScaling();

  // Input parameters
  double lambda_;
  BlockMatrix A_;
  BlockMatrix AT_;
  Expression f_expr_;

  // Prox function
  std::unique_ptr<ProxOperator> prox_;

  // Preprocessed parameters, f(x) = alpha*g(x) + c'x
  const Expression* g_expr_;
  double alpha_;
  BlockVector c_;
};

// Preprocess f(x) to extract f(x) = alpha*g(Ax + b) + c'x.
// NOTE(mwytock): We assume the input expression has already undergone
// processing and thus we dont need to handle fully general expressions here.
void ProxBlockVectorOperator::Preprocess() {
  // Add affine term
  g_expr_ = &f_expr_;
  if (g_expr_->expression_type() == Expression::ADD) {
    CHECK_EQ(2, f_expr_.arg_size());
    g_expr_ = &f_expr_.arg(0);

    // Extract linear function
    BlockMatrix A;
    BlockVector b;
    affine::BuildAffineOperator(f_expr_.arg(1), "c", &A, &b);
    for (const std::string& var_id : A.col_keys())
      c_(var_id) = ToVector(A("c", var_id).impl().AsDense());
  }

  // Multiply scalar constant
  if (g_expr_->expression_type() == Expression::MULTIPLY &&
      IsScalarConstant(g_expr_->arg(0))) {
    alpha_ = GetScalarConstant(g_expr_->arg(0));
    g_expr_ = &g_expr_->arg(1);
  } else {
    alpha_ = 1;
  }

  VLOG(2) << "Preprocess, alpha = " << alpha_ << ", c = " << c_.DebugString();
}

void ProxBlockVectorOperator::InitScaling() {
  BlockMatrix ATA = AT_*A_;
  VLOG(1) << "ATA: " << ATA.DebugString();
  // for (const auto& col_iter : ATA.data()) {
  //   CHECK(col_iter.second.size() == 1 &&
  //         col_iter.first == col_iter.second.begin()->first)
  //       << "Trying to invert non block diagonal matrix\n" << DebugString();
  //   const std::string& key = col_iter.first;
}

std::unique_ptr<BlockVectorOperator> CreateProxOperator(
    double lambda,
    BlockMatrix A,
    const Expression& f_expr) {
  return std::unique_ptr<BlockVectorOperator>(new ProxBlockVectorOperator(
      lambda, A, f_expr));
}
