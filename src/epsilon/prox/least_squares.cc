
#include "epsilon/affine/affine.h"
#include "epsilon/affine/affine_matrix.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

// TODO(mwytock): Likely want to unify this with LeastSquaresProx. Perhaps we
// can just get rid of the old implementation when affine_matrix.cc is a little
// more mature.
//
// lam*||AX + B||_F^2 with dense A
// Expression tree:
// SUM
//   POWER (p: 2)
//     AFFINE (X)
class LeastSquaresMatrixProx final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();

    affine::MatrixOperator op = affine::BuildMatrixOperator(
        arg.f_expr().arg(0).arg(0));
    CHECK(op.B.isIdentity());
    A_ = op.A;
    ATB_ = A_.transpose()*op.C;

    const int m = A_.rows();
    const int n = A_.cols();
    if (m <= n) {
      VLOG(2) << "Computing I/lambda + AA' (" << m << ", " << m << ")";
      solver_.compute(
          Eigen::MatrixXd::Identity(m, m)/lambda_/2 + A_*A_.transpose());
    } else {
      VLOG(2) << "Computing I + lambda*AA' (" << n << ", " << n << ")";
      solver_.compute(
          Eigen::MatrixXd::Identity(n, n) + 2*arg.lambda()*A_.transpose()*A_);
    }
    CHECK_EQ(solver_.info(), Eigen::Success);
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    const int m = ATB_.rows();
    const int n = ATB_.cols();

    Eigen::MatrixXd Q = ToMatrix(v, m, n) - 2*lambda_*ATB_;
    if (A_.rows() <= A_.cols()) {
      return ToVector(Q - A_.transpose()*solver_.solve(A_*Q));
    } else {
      return ToVector(solver_.solve(Q));
    }
  }

private:
  Eigen::MatrixXd A_, ATB_;
  double lambda_;
  Eigen::LLT<Eigen::MatrixXd> solver_;
};
REGISTER_PROX_OPERATOR(LeastSquaresMatrixProx);

// lam*||Ax + b||_2^2 with dense or sparse A
// Expression tree:
// SUM
//   POWER (p: 2)
//     AFFINE (x)
class LeastSquaresProx final : public ProxOperator {
 public:
  void Init(const ProxOperatorArg& arg) override {
    const Expression& expr_arg = arg.f_expr().arg(0).arg(0);

    // Get affine function
    const int m = GetDimension(expr_arg);
    const int n = arg.var_map().n();
    DynamicMatrix A = DynamicMatrix::Zero(m, n);
    DynamicMatrix b0 = DynamicMatrix::FromDense(Eigen::VectorXd::Zero(m));
    BuildAffineOperator(expr_arg, arg.var_map(), &A, &b0);
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
REGISTER_PROX_OPERATOR(LeastSquaresProx);
