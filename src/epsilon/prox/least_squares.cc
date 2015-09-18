
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

// lam*||Ax + b||_2^2 with dense or sparse A
class LeastSquaresProx final : public ProxOperator {
 protected:
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // SUM
    //   POWER (p: 2)
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
REGISTER_PROX_OPERATOR(LeastSquaresProx);
