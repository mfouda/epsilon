
#include "epsilon/affine/affine.h"
#include "epsilon/affine/affine_matrix.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"

// I(Ax == b)
// Expression tree:
// INDICATOR (cone: ZERO)
//   AFFINE (Ax - b)

class LinearEqualityProx final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    graph_form_ = InitGraph(arg);
    if (graph_form_)
      return;

    matrix_form_ = InitMatrix(arg);
    if (matrix_form_)
      return;

    InitGeneric(arg);
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    if (graph_form_)
      return ApplyGraph(v);
    if (matrix_form_)
      return ApplyMatrix(v);
    return ApplyGeneric(v);
  }

private:
  // I(Y == AX + BY) or I(Y == XA + B)
  bool InitGraph(const ProxOperatorArg& arg) {
    CHECK_EQ(1, arg.f_expr().arg_size());
    if (arg.f_expr().arg(0).expression_type() != Expression::ADD ||
        arg.f_expr().arg(0).arg_size() != 2) {
      return false;
    }

    bool negate = false;
    const Expression* AX = nullptr;
    const Expression* Y = nullptr;
    for (int i = 0; i < 2; i++) {
      const Expression& argi = arg.f_expr().arg(0).arg(i);

      if (argi.expression_type() == Expression::VARIABLE) {
        AX = &arg.f_expr().arg(0).arg(1-i);
        Y  = &arg.f_expr().arg(0).arg(i);
        break;
      } else if (argi.expression_type() == Expression::NEGATE &&
                 argi.arg(0).expression_type() == Expression::VARIABLE) {
        negate = true;
        AX = &arg.f_expr().arg(0).arg(1-i);
        Y  = &arg.f_expr().arg(0).arg(i).arg(0);
        break;
      }
    }
    if (Y == nullptr)
      return false;

    VariableSet vars = GetVariables(*AX);
    if (vars.size() != 1)
      return false;
    const Expression* X = *vars.begin();

    affine::MatrixOperator op = affine::BuildMatrixOperator(*AX);

    int m, n, k;
    if (op.B.isIdentity()) {
      // I(AX + B == Y)
      const_lhs_ = true;
      m = op.A.rows();
      n = op.A.cols();
      k = op.B.cols();
      A_ = op.A;
      B_ = op.C.rows() ? op.C : Eigen::MatrixXd::Zero(m, k);
    } else if (op.A.isIdentity()) {
      // I(XA + B == Y)
      const_lhs_ = false;
      m = op.B.rows();
      n = op.B.cols();
      k = op.A.rows();
      A_ = op.B;
      B_ = op.C.rows() ? op.C : Eigen::MatrixXd::Zero(k, n);
    } else {
      return false;
    }

    if (!negate) {
      A_ *= -1;
      B_ *= -1;
    }

    if (m <= n) {
      llt_solver_.compute(A_*A_.transpose() + Eigen::MatrixXd::Identity(m, m));
    } else {
      llt_solver_.compute(A_.transpose()*A_ + Eigen::MatrixXd::Identity(n, n));
    }
    CHECK_EQ(llt_solver_.info(), Eigen::Success);

    VLOG(2) << "InitGraph m=" << m << ", n=" << n << ", k=" << k
            << ", const_lhs=" << const_lhs_
            << "\nA:\n" << MatrixDebugString(A_)
            << "\nB:\n" << MatrixDebugString(B_);

    X_index_ = arg.var_map().Get(X->variable().variable_id());
    Y_index_ = arg.var_map().Get(Y->variable().variable_id());
    X_m_ = GetDimension(*X, 0);
    X_n_ = GetDimension(*X, 1);
    Y_m_ = GetDimension(*Y, 0);
    Y_n_ = GetDimension(*Y, 1);
    return true;
  }

  // I(AX + B == 0) or I(XA + B == 0)
  bool InitMatrix(const ProxOperatorArg& arg) {
    CHECK_EQ(1, arg.f_expr().arg_size());
    const Expression* AX = &arg.f_expr().arg(0);

    VariableSet vars = GetVariables(arg.f_expr().arg(0));
    if (vars.size() != 1)
      return false;
    const Expression* X = *vars.begin();

    const int m = GetDimension(*AX, 0);
    const int n = GetDimension(*AX, 1);
    affine::MatrixOperator op = affine::BuildMatrixOperator(*AX);
    if (op.B.isIdentity()) {
      const_lhs_ = true;
      A_ = op.A;
      B_ = op.C.rows() ? op.C : Eigen::MatrixXd::Zero(m, n);
    } else if (op.A.isIdentity()) {
      const_lhs_ = false;
      A_ = op.B.transpose();
      B_ = op.C.rows() ? op.C.transpose().eval() : Eigen::MatrixXd::Zero(n, m);
    } else {
      return false;
    }

    svd_solver_.compute(A_, Eigen::ComputeThinU|Eigen::ComputeThinV);
    X_m_ = GetDimension(*X, 0);
    X_n_ = GetDimension(*X, 1);
    return true;
  }

  // I(Ax + b == 0)
  void InitGeneric(const ProxOperatorArg& arg) {
    LOG(FATAL) << "Not implemented";

    // const int m = GetDimension(arg.f_expr().arg(0));
    // const int n = arg.var_map().n();
    // DynamicMatrix A = DynamicMatrix::Zero(m, n);
    // DynamicMatrix b = DynamicMatrix::Zero(m, 1);
    // BuildAffineOperator(arg.f_expr().arg(0), arg.var_map(), &A, &b);
    // CHECK(m <= n) << "m > n not implemented";

    // VLOG(1) << "Factoring AA', A = " << A.rows() << " x " << A.cols();
    // A_ = A.AsDense();
    // b_ = -b.AsDense();
    // AAT_solver_.compute(A_*A_.transpose());
    // CHECK_EQ(AAT_solver_.info(), Eigen::Success);
  }

  Eigen::VectorXd ApplyGraph(const Eigen::VectorXd& vu) {
    Eigen::MatrixXd V = ToMatrix(vu.segment(X_index_, X_m_*X_n_), X_m_, X_n_);
    Eigen::MatrixXd U = ToMatrix(vu.segment(Y_index_, Y_m_*Y_n_), Y_m_, Y_n_);

    Eigen::MatrixXd X, Y;
    if (const_lhs_) {
      if (A_.rows() <= A_.cols()) {
        Eigen::MatrixXd Z = llt_solver_.solve(A_*V - U + B_);
        X = V - A_.transpose()*Z;
        Y = U + Z;
      } else {
        X = llt_solver_.solve(V + A_.transpose()*(U - B_));
        Y = A_*X + B_;
      }
    } else {
      if (A_.rows() <= A_.cols()) {
        X = llt_solver_.solve(
            A_*((U - B_).transpose()) + V.transpose()).transpose();
        Y = X*A_ + B_;
      } else {
        Eigen::MatrixXd Z = llt_solver_.solve(
            (V*A_ - U + B_).transpose()).transpose();
        X = V - Z*A_.transpose();
        Y = U + Z;
      }
    }

    Eigen::VectorXd xy(X_m_*X_n_ + Y_m_*Y_n_);
    xy.segment(X_index_, X_m_*X_n_) = ToVector(X);
    xy.segment(Y_index_, Y_m_*Y_n_) = ToVector(Y);
    return xy;
  }

  Eigen::VectorXd ApplyMatrix(const Eigen::VectorXd& v) {
    Eigen::MatrixXd V = ToMatrix(v, X_m_, X_n_);
    if (!const_lhs_) V.transposeInPlace();
    Eigen::MatrixXd X = V - svd_solver_.solve(A_*V + B_);
    if (!const_lhs_) X.transposeInPlace();
    return ToVector(X);
  }

  Eigen::VectorXd ApplyGeneric(const Eigen::VectorXd& v) {
    LOG(FATAL) << "Not implemented";
  }

  // Solver decision tree
  bool matrix_form_, graph_form_, const_lhs_;

  // For matrix-based methods
  int X_index_, X_m_, X_n_;
  int Y_index_, Y_m_, Y_n_;

  // For generic vector method

  // All methods use these
  Eigen::MatrixXd A_;
  Eigen::MatrixXd B_;
  Eigen::LLT<Eigen::MatrixXd> llt_solver_;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd_solver_;
};
REGISTER_PROX_OPERATOR(LinearEqualityProx);
