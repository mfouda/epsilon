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
  std::string,
  std::function<std::unique_ptr<ProxOperator>()>>* kProxOperatorMap;

std::unordered_map<
  std::string,
  std::function<std::unique_ptr<BlockProxOperator>()>>* kBlockProxOperatorMap;

class ProxBlockVectorOperator final : public BlockVectorOperator {
 public:
  ProxBlockVectorOperator(
      double lambda,
      BlockMatrix A,
      const Expression& f_expr)
      : lambda_(lambda),
      A_(A),
      f_expr_(f_expr) {
    CHECK(kBlockProxOperatorMap != nullptr) << "No registered operators";
    CHECK(kProxOperatorMap != nullptr) << "No registered operators";
  }

  void Init() override {
    AT_ = A_.Transpose();

    Preprocess();
    InitScaling();

    auto iter = kBlockProxOperatorMap->find(g_expr_->proximal_operator().name());
    if (iter != kBlockProxOperatorMap->end()) {
      // New style block prox
      block_vector_prox_ = true;
      prox_ = iter->second();
      prox_->Init(ProxOperatorArg(alpha_*lambda_, &A_, g_expr_, nullptr));
    } else {
      // Legacy vector prox
      block_vector_prox_ = false;
      auto iter2 = kProxOperatorMap->find(g_expr_->proximal_operator().name());
      if (iter2 == kProxOperatorMap->end()) {
        LOG(FATAL) << "No proximal operator for "
                   << g_expr_->proximal_operator().name() << "\n"
                   << g_expr_->DebugString();
      }
      legacy_prox_ = iter2->second();
      var_map_.Insert(f_expr_);
      legacy_prox_->Init(ProxOperatorArg(alpha_*lambda_, nullptr, g_expr_, &var_map_));
    }
  }

  virtual BlockVector Apply(const BlockVector& v) override {
    BlockVector v_c = AT_*v - lambda_*c_;
    if (block_vector_prox_)
      return prox_->Apply(v_c);

    // Old style prox. First we have to extract vectors from BlockVector
    Eigen::VectorXd v_vec = Eigen::VectorXd::Zero(var_map_.n());
    for (auto iter : var_map_.offsets()) {
      if (v_c.has_key(iter.first))
        v_vec.segment(iter.second, var_map_.Size(iter.first)) = v_c(iter.first);
    }

    legacy_prox_->Apply(v_vec);
    Eigen::VectorXd x_vec = legacy_prox_->Apply(v_vec);
    BlockVector x;
    for (auto iter : var_map_.offsets()) {
      x(iter.first) = x_vec.segment(iter.second, var_map_.Size(iter.first));
    }
    return x;
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
  std::unique_ptr<BlockProxOperator> prox_;

  // Legacy prox stuff
  bool block_vector_prox_;
  VariableOffsetMap var_map_;
  std::unique_ptr<ProxOperator> legacy_prox_;

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
