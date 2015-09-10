#include "epsilon/operators/prox.h"

#include <glog/logging.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

#include "epsilon/expression/expression_util.h"
#include "epsilon/operators/affine.h"

// Arguments to the proximal operator, see below
class ProxOperatorArg {
 public:
  int n() const { return n_; };
  double lambda() const { return lambda_; };

  const DynamicMatrix& A() const { return *A_; }
  const DynamicMatrix& b() const { return *b_; }

 private:
  double lambda_;

  // Not owned by us
  const DynamicMatrix* A_;
  const DynamicMatrix* b_;
};

// A general interface to a proximal operator for a function, f : R^n -> R, of
// the form: lambda*f(Ax + b)
//
// Many operators require special structure in A, e.g. diagonal or identity.
//
// TODO(mwytock): Figure out more precisely the interface for matrix
// variables.
class ProxOperator : public VectorOperator {
 public:
  void Init() override {
    // lambda_ *= f_.alpha();
    // A_ = DynamicMatrix::Zero(m(), n());
    // b_ = DynamicMatrix::FromDense(Eigen::VectorXd::Zero(m()));

    // // TODO(mwytock): BuildAffineOperator() should likely take
    // // VariableOffsetsMap as input, not just n.
    // CHECK_EQ(1, f_.arg_size());
    // BuildAffineOperator(f_.arg(0), &A_, &b_);

    // VLOG(2) << "ProxOperator " << ProxFunction::Function_Name(f_.function())
    //         << "\nA:\n" << A_.DebugString()
    //         << "\nb: "  << b_.DebugString();
    // InitFunction();
    // VLOG(2) << "Init done";

    // if (f_.has_affine()) {
    //   DynamicMatrix cT(1, n()), c0(1, 1);
    //   BuildAffineOperator(f_.affine(), &cT, &c0);
    //   c_ = cT.AsDense().transpose();
    // }
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    const Eigen::VectorXd* input = &v;

    // Eigen::VectorXd v_copy;
    // if (f_.has_affine()) {
    //   v_copy = *input - lambda()*c_;
    //   input = &v_copy;
    // }

    return ApplyFunction(*input);
  }

 protected:
  // To be implemented by sub classes
  virtual void InitFunction(const ProxOperatorArg& arg) {}
  virtual Eigen::VectorXd ApplyFunction(const Eigen::VectorXd& v) = 0;

  // // Accessors for sub classes
  // // TODO(mwytock): These should probably be broken out into InitFunctionInput
  // // and ApplyFunctionInput structures
  // double lambda() const { return lambda_; }
  // int m() const { return GetDimension(f_.arg(0)); }
  // int n() const { return n_; }
  // int arg_m() const { return GetDimension(f_.arg(0), 0); }
  // int arg_n() const { return GetDimension(f_.arg(0), 1); }
  const DynamicMatrix& arg_A() const { return A_; }
  const DynamicMatrix& arg_b() const { return b_; }

 private:
  friend std::unique_ptr<VectorOperator> CreateProxOperator(
      const Expression& expr, double lambda);

  Expression expr_;
  double lambda_;

  // Affine transformation of function arg
  DynamicMatrix A_, b_;

  // Affine addition
  Eigen::VectorXd c_;
};

// class SumSquaresProx final : public ProxOperator {
//  public:
//   void InitFunction() override {
//     CHECK(!arg_b().is_sparse());

//     if (arg_A().is_sparse()) {
//       // Only special case of diagonal is supported for sparse matrices for
//       // now.
//       const SparseXd& A = arg_A().sparse();
//       Atb_ = A.transpose()*arg_b().dense();
//       if (m() <= n()) {
//         SparseXd AAt = A*A.transpose();
//         CHECK(IsDiagonal(AAt));
//         D_.resize(m());
//         D_.diagonal() = AAt.diagonal().eval().array() + 0.5/lambda();
//       } else {
//         SparseXd AtA = A.transpose()*A;
//         CHECK(IsDiagonal(AtA));
//         D_.resize(n());
//         D_.diagonal() = 2*lambda()*AtA.diagonal().eval().array() + 1;
//       }
//       D_ = D_.inverse();
//     } else {
//       const Eigen::MatrixXd& A = arg_A().dense();
//       Atb_ = A.transpose()*arg_b().dense();
//       if (m() <= n()) {
//         VLOG(2) << "Computing I/lambda + AA' (" << m() << ", " << m() << ")";
//         solver_.compute(
//             Eigen::MatrixXd::Identity(m(), m())/lambda()/2 + A*A.transpose());
//       } else {
//         VLOG(2) << "Computing I + lambda*AA' (" << n() << ", " << n() << ")";
//         solver_.compute(
//           Eigen::MatrixXd::Identity(n(), n()) + 2*lambda()*A.transpose()*A);
//       }
//       CHECK_EQ(solver_.info(), Eigen::Success);
//     }
//   }

//   Eigen::VectorXd ApplyFunction(const Eigen::VectorXd& v) override {
//     Eigen::VectorXd q = v - 2*lambda()*Atb_;
//     if (arg_A().is_sparse()) {
//       const SparseXd& A = arg_A().sparse();
//       if (m() <= n()) {
//         return q - A.transpose()*D_*A*q;
//       } else {
//         return D_*q;
//       }
//     } else {
//       const Eigen::MatrixXd& A = arg_A().dense();
//       if (m() <= n()) {
//         return q - A.transpose()*solver_.solve(A*q);
//       } else {
//         return solver_.solve(q);
//       }
//     }
//   }

//  private:
//   Eigen::VectorXd Atb_;
//   Eigen::LLT<Eigen::MatrixXd> solver_;
//   Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> D_;
// };

class NormL1Prox final : public ProxOperator {
 public:
  void InitFunction(const ProxOperatorArg& arg) override {
    CHECK(arg.b().is_zero());
    CHECK(arg.A().is_sparse());
    CHECK(IsDiagonal(arg.A().sparse()));
    lambda_elemwise_ = arg.lambda()*arg.A().sparse().diagonal();
  }

  Eigen::VectorXd ApplyFunction(const Eigen::VectorXd& v) override {
    return (( v.array()-lambda_elemwise_.array()).max(0) -
            (-v.array()-lambda_elemwise_.array()).max(0));
  }

 private:
  // Lambda applied elementwise
  Eigen::VectorXd lambda_elemwise_;
};

// // sum of L2 norm applied to rows of matrix variable
// class NormL1L2Prox final : public ProxOperator {
//   Eigen::VectorXd ApplyFunction(const Eigen::VectorXd& v) override {
//     // TODO(mwytock): Need to use elementwise weighting here
//     Eigen::MatrixXd V = ToMatrix(v, arg_m(), arg_n());
//     Eigen::VectorXd norms = V.array().square().rowwise().sum().sqrt();
//     Eigen::VectorXd s = 1 - (norms.array()/lambda()).max(1).inverse();
//     return ToVector(s.asDiagonal()*V);
//   }
// };

// class NegativeLogDetProx final : public ProxOperator {
//   void InitFunction() override {
//     CHECK_EQ(arg_m(), arg_n());
//   }

//   Eigen::VectorXd ApplyFunction(const Eigen::VectorXd& v) override {
//     Eigen::MatrixXd V = ToMatrix(v, arg_m(), arg_n());
//     Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(V);
//     CHECK_EQ(solver.info(), Eigen::Success);
//     const Eigen::VectorXd& d = solver.eigenvalues();
//     const Eigen::MatrixXd& U = solver.eigenvectors();

//     Eigen::VectorXd x_tilde =
//         (d.array() + (d.array().square() + 4*lambda()).sqrt())/2;
//     Eigen::MatrixXd X = U*x_tilde.asDiagonal()*U.transpose();

//     return ToVector(X);
//   }
// };

// // Ax = b
// class IndicatorZeroProx final : public ProxOperator {
//   void InitFunction() override {
//     if (arg_A().is_sparse()) {
//       sp_solver_.compute(arg_A().sparse()*arg_A().sparse().transpose());
//       CHECK_EQ(sp_solver_.info(), Eigen::Success);
//     } else {
//       solver_.compute(arg_A().dense()*arg_A().dense().transpose());
//       CHECK_EQ(solver_.info(), Eigen::Success);
//     }

//     CHECK(!arg_b().is_sparse());
//   }

//   Eigen::VectorXd ApplyFunction(const Eigen::VectorXd& v) override {
//     const Eigen::VectorXd& b = arg_b().dense();

//     if (arg_A().is_sparse()) {
//       const SparseXd& A = arg_A().sparse();
//       Eigen::VectorXd u = sp_solver_.solve(A*v + b);
//       return v - A.transpose()*u;
//     } else {
//       const Eigen::MatrixXd& A = arg_A().dense();
//       return v - A.transpose()*solver_.solve(A*v + b);
//     }
//   }

// private:
//   Eigen::LLT<Eigen::MatrixXd> solver_;
//   Eigen::SimplicialLDLT<SparseXd> sp_solver_;
// };

// Rules for matching proximal operators to expression trees
struct ProxOperatorRule {
  std::function<bool(const Expression& expr)> match;
  std::function<std::unique_ptr<ProxOperator>()> create;
};

#define PROX_RULE(op, match_expr)                    \
  {                                                  \
    [] (const Expression& expr) -> bool {            \
      return (match_expr);                           \
    },                                               \
    [] () -> std::unique_ptr<ProxOperator> {         \
      return std::unique_ptr<ProxOperator>(new op);  \
    }                                                \
  }

std::vector<ProxOperatorRule> kProxOperatorRules = {
  PROX_RULE(NormL1Prox,
            expr.expression_type() == Expression::NORM_P &&
            expr.p() == 1),
};

// std::vector<ProxOperatorRule> kProxOperatorRules = {
//   PROX_RULE(LeastSquaresProx,
//             expr.expression_type() == Expression::POWER,
//             expr.p() == 2 &&
//             expr.arg(0).expression_type() == Expression::NORM_P &&
//             expr.arg(0).p() == 2),
//   PROX_RULE(NegativeLogDetProx,
//             expr.expression_type() == Expression::NEGATE &&
//             expr.arg().expression_type() == Expression::LOG_DET),
//   PROX_RULE(NormL1Prox,
//             expr.expression_type() == Expression::NORM_P &&
//             expr.p() == 1),
//   PROX_RULE(NormL2L1Prox,
//             expr.expression_type() == Expression::NORM_PQ &&
//             expr.p() == 2,
//             expr.q() == 1),
// };

std::unique_ptr<VectorOperator> CreateProxOperator(
    const Expression& expr, double lambda) {
  VLOG(2) << "CreateProxOperator\n" << expr.DebugString();

  // TODO(mwytock): Preprocess expression to handle affine addition

  std::unique_ptr<ProxOperator> op;
  for (const ProxOperatorRule& rule : kProxOperatorRules) {
    if (rule.match(expr)) {
      op = rule.create();
      break;
    }
  }

  op->expr_ = expr;
  op->lambda_ = lambda;
  return std::move(op);
}
