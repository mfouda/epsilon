#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"

// I(Ax == b)
class LinearEqualityProx final : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // INDICATOR (cone: ZERO)
    //   AFFINE (Ax - b)

    const int m = GetDimension(arg.f_expr().arg(0));
    const int n = arg.var_map().n();
    DynamicMatrix A = DynamicMatrix::Zero(m, n);
    DynamicMatrix b = DynamicMatrix::Zero(m, 1);
    BuildAffineOperator(arg.f_expr().arg(0), arg.var_map(), &A, &b);

    CHECK(!A.is_sparse()) << "Sparse A not implemented";
    CHECK(m <= n) << "m > n not implemented";

    A_ = A.dense();
    b_ = -b.AsDense();
    AAT_solver_.compute(A_*A_.transpose());
    CHECK_EQ(AAT_solver_.info(), Eigen::Success);
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    return v - A_.transpose()*(AAT_solver_.solve(A_*v - b_));
  }

private:
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  Eigen::LLT<Eigen::MatrixXd> AAT_solver_;
};
REGISTER_PROX_OPERATOR(LinearEqualityProx);
