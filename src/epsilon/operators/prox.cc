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

#include "epsilon/operators/prox.h"

#include <glog/logging.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

#include "epsilon/expression/expression_util.h"
#include "epsilon/operators/affine.h"

// TODO(mwytock): Utility functions that should likely go elsewhere
SparseXd GetSparseOperator(
    const Expression& expr,
    const VariableOffsetMap& var_map) {
  const int m = GetDimension(expr);
  const int n = var_map.n();
  DynamicMatrix A = DynamicMatrix::Zero(m, n);
  DynamicMatrix b = DynamicMatrix::FromDense(Eigen::VectorXd::Zero(m));
  BuildAffineOperator(expr, var_map, &A, &b);

  CHECK(b.is_zero());
  CHECK(A.is_sparse());
  return A.sparse();
}

Eigen::VectorXd GetElementwiseMultiply(
    const Expression& expr,
    const VariableOffsetMap& var_map) {
  SparseXd A = GetSparseOperator(expr, var_map);
  CHECK(IsDiagonal(A));
  return A.diagonal();
}

bool IsEpigraphForm(const Expression& expr) {
  return (expr.expression_type() == Expression::INDICATOR &&
          expr.cone().cone_type() == Cone::NON_NEGATIVE &&
          expr.arg_size() == 2);
}

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
    g_prox_->Init(ProxOperatorArg(alpha_*lambda_, g_expr_, &var_map_));
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

// lam*||Ax + b||_2^2 with dense or sparse A
class LeastSquaresProx final : public ProxOperator {
 protected:
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // POWER (p: 2)
    //   NORM_P (p: 2)
    //     AFFINE (x)

    // Get affine function
    const int m = GetDimension(arg.f_expr().arg(0).arg(0));
    const int n = arg.var_map().n();
    DynamicMatrix A = DynamicMatrix::Zero(m, n);
    DynamicMatrix b0 = DynamicMatrix::FromDense(Eigen::VectorXd::Zero(m));
    BuildAffineOperator(arg.f_expr().arg(0).arg(0), arg.var_map(), &A, &b0);
    if (A.is_sparse()) {
      sparse_ = true;
      A_sparse_ = A.sparse();
    } else {
      sparse_ = false;
      A_dense_ = A.dense();
    }
    Eigen::VectorXd b = b0.dense();

    if (sparse_) {
      // TODO(mwytock): Support general sparse matrices. Currently only diagonal
      // sparse matrices are supported
      const SparseXd& A = A_sparse_;
      Atb_ = A.transpose()*b;
      if (A.rows() <= A.cols()) {
        SparseXd AAt = A*A.transpose();
        CHECK(IsDiagonal(AAt));
        D_.resize(m);
        D_.diagonal() = AAt.diagonal().eval().array() + 0.5/arg.lambda();
      } else {
        SparseXd AtA = A.transpose()*A;
        CHECK(IsDiagonal(AtA));
        D_.resize(n);
        D_.diagonal() = 2*arg.lambda()*AtA.diagonal().eval().array() + 1;
      }
      D_ = D_.inverse();
    } else {
      const Eigen::MatrixXd& A = A_dense_;
      Atb_ = A.transpose()*b;
      if (A.rows() <= A.cols()) {
        VLOG(2) << "Computing I/lambda + AA' (" << m << ", " << m << ")";
        solver_.compute(
            Eigen::MatrixXd::Identity(m, m)/arg.lambda()/2 + A*A.transpose());
      } else {
        VLOG(2) << "Computing I + lambda*AA' (" << n << ", " << n << ")";
        solver_.compute(
          Eigen::MatrixXd::Identity(n, n) + 2*arg.lambda()*A.transpose()*A);
      }
      CHECK_EQ(solver_.info(), Eigen::Success);
    }

    lambda_ = arg.lambda();
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    Eigen::VectorXd q = v - 2*lambda_*Atb_;
    if (sparse_) {
      const SparseXd& A = A_sparse_;
      if (A.rows() <= A.cols()) {
        return q - A.transpose()*D_*A*q;
      } else {
        return D_*q;
      }
    } else {
      const Eigen::MatrixXd& A = A_dense_;
      if (A.rows() <= A.cols()) {
        return q - A.transpose()*solver_.solve(A*q);
      } else {
        return solver_.solve(q);
      }
    }
  }

 private:
  double lambda_;
  bool sparse_;
  Eigen::MatrixXd A_dense_;
  SparseXd A_sparse_;
  Eigen::VectorXd Atb_;
  Eigen::LLT<Eigen::MatrixXd> solver_;
  Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> D_;
};


// lam*||a .* x||_1
class NormL1Prox final : public ProxOperator {
 public:
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // NORM_P (p: 1)
    //   VARIABLE (x)
    t_ = arg.lambda()*GetElementwiseMultiply(
        arg.f_expr().arg(0), arg.var_map());
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    return (( v.array()-t_.array()).max(0) -
            (-v.array()-t_.array()).max(0));
  }

 private:
  // Elementwise soft-thresholding values
  Eigen::VectorXd t_;
};

// lam*||x||_2
class NormL2Prox final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // NORM_P (p: 2)
    //   VARIABLE (x)
    lambda_ = arg.lambda();
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    const double v_norm = v.norm();
    if (v_norm >= lambda_) {
      return (1 - lambda_/v_norm)*v;
    } else {
      return v;
    }
  }

private:
  double lambda_;
};

// I(||x||_2 <= t)
class NormL2Epigraph final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // INDICATOR (cone: NON_NEGATIVE)
    //   VARIABLE (t)
    //   NORM_P (p: 2)
    //     VARIABLE (x)
    A_ = GetSparseOperator(arg.f_expr().arg(0), arg.var_map());
    B_ = GetSparseOperator(arg.f_expr().arg(1).arg(0), arg.var_map());
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& vs) override {
    Eigen::VectorXd v = B_*vs;
    const double s = (A_*vs)(0);
    const double v_norm = v.norm();

    if (v_norm <= -s) {
      return Eigen::VectorXd::Zero(vs.rows());
    } else if (v_norm <= s) {
      return vs;
    } else {
      return 0.5*(1 + s/v_norm)*(
          B_.transpose()*v +
          static_cast<Eigen::MatrixXd>(A_.transpose())*v_norm);
    }
  }

private:
  SparseXd A_, B_;
};

// lam*||X||_{1,2}
class NormL1L2Prox final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // NORM_PQ (p: 1, q: 2)
    //   VARIABLE (X)
    const Expression& var_expr = arg.f_expr().arg(0);
    CHECK_EQ(var_expr.expression_type(), Expression::VARIABLE);
    m_ = GetDimension(var_expr, 0);
    n_ = GetDimension(var_expr, 1);
    lambda_ = arg.lambda();
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    Eigen::MatrixXd V = ToMatrix(v, m_, n_);
    Eigen::VectorXd norms = V.array().square().rowwise().sum().sqrt();
    Eigen::VectorXd s = 1 - (norms.array()/lambda_).max(1).inverse();
    return ToVector(s.asDiagonal()*V);
  }

private:
  double lambda_;
  int m_, n_;
};

// -lam*log|X|
class NegativeLogDetProx final : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // NEGATE
    //   LOG_DET
    //     VARIABLE (X)
    const Expression& var_expr = arg.f_expr().arg(0).arg(0);
    CHECK_EQ(var_expr.expression_type(), Expression::VARIABLE);
    CHECK_EQ(GetDimension(var_expr, 0),
             GetDimension(var_expr, 1));
    n_ = GetDimension(var_expr, 0);
    lambda_ = arg.lambda();
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    Eigen::MatrixXd V = ToMatrix(v, n_, n_);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(V);
    CHECK_EQ(solver.info(), Eigen::Success);
    const Eigen::VectorXd& d = solver.eigenvalues();
    const Eigen::MatrixXd& U = solver.eigenvectors();

    Eigen::VectorXd x_tilde =
        (d.array() + (d.array().square() + 4*lambda_).sqrt())/2;
    Eigen::MatrixXd X = U*x_tilde.asDiagonal()*U.transpose();

    return ToVector(X);
  }

private:
  double lambda_;
  int n_;
};

// Rules for matching proximal operators to expression trees
struct ProxOperatorRule {
  std::function<bool(const Expression& expr)> match;
  std::function<std::unique_ptr<ProxOperator>()> create;
};

#define PROX_RULE(op, match_expr)                       \
  {                                                     \
    [] (const Expression& expr) -> bool {               \
      return (match_expr);                              \
    },                                                  \
    [] () -> std::unique_ptr<ProxOperator> {            \
      return std::unique_ptr<ProxOperator>(new op);     \
    }                                                   \
  }

std::vector<ProxOperatorRule> kProxOperatorRules = {
  PROX_RULE(LeastSquaresProx,
            expr.expression_type() == Expression::POWER &&
            expr.p() == 2 &&
            expr.arg(0).expression_type() == Expression::NORM_P &&
            expr.arg(0).p() == 2),
  PROX_RULE(NegativeLogDetProx,
            expr.expression_type() == Expression::NEGATE &&
            expr.arg(0).expression_type() == Expression::LOG_DET),
  PROX_RULE(NormL1Prox,
            expr.expression_type() == Expression::NORM_P &&
            expr.p() == 1),
  PROX_RULE(NormL1L2Prox,
            expr.expression_type() == Expression::NORM_PQ &&
            expr.p() == 1 &&
            expr.q() == 2),
  PROX_RULE(NormL2Epigraph,
            IsEpigraphForm(expr) &&
            expr.arg(1).expression_type() == Expression::NORM_P &&
            expr.arg(1).p() == 2),
  PROX_RULE(NormL2Prox,
            expr.expression_type() == Expression::NORM_P &&
            expr.p() == 2),

};

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
  for (const ProxOperatorRule& rule : kProxOperatorRules) {
    if (rule.match(*g_expr_)) {
      g_prox_ = rule.create();
      break;
    }
  }

  CHECK(g_prox_.get() !=nullptr) << "No rule\n" << g_expr_->DebugString();
}

std::unique_ptr<VectorOperator> CreateProxOperator(
    double lambda,
    const Expression& f_expr,
    const VariableOffsetMap& var_map) {
  return std::unique_ptr<VectorOperator>(new ProxVectorOperator(
      lambda, f_expr, var_map));
}
