
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/linear/linear_map.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/block_matrix.h"

// I(Ax + b = 0)
// Expression tree:
// INDICATOR (cone: ZERO)
//   AFFINE
class LinearEqualityProx final : public BlockProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    BlockMatrix A;
    BlockVector b;
    affine::BuildAffineOperator(arg.f_expr().arg(0), "f", &A, &b);

    graph_form_ = InitGraphForm(A, b);
    if (graph_form_)
      return;

    // Standard case, solve:
    // x  + A'y = v
    // Ax       = -b
    A_ = A;
    AT_ = A.Transpose();
    AAT_inv_ = (A_*AT_).Inverse();
    q_ = AT_*(AAT_inv_*b);
  }

  BlockVector Apply(const BlockVector& v) override {
    if (graph_form_) {
      Eigen::VectorXd v_y = (
          v.has_key(y_) ? v(y_) : Eigen::VectorXd::Zero(B_.impl().m()));
      Eigen::VectorXd v_x = (
          v.has_key(y_) ? v(x_) : Eigen::VectorXd::Zero(B_.impl().n()));

      BlockVector x;
      Eigen::VectorXd z = alpha_*v_y + c_;
      Eigen::VectorXd w = alpha_*alpha_*v_x - BT_*z;
      x(x_) = BTB_inv_*w;
      x(y_) = -(1/alpha_)*(B_*x(x_) + c_);
      return x;
    }

    return v - AT_*(AAT_inv_*(A_*v)) - q_;
  }

private:
  bool InitGraphForm(const BlockMatrix& A, const BlockVector& b) {
    std::set<std::string> col_keys_set = A.col_keys();
    std::vector<std::string> col_keys{col_keys_set.begin(), col_keys_set.end()};
    if (col_keys.size() != 2)
      return false;

    bool has_scalar_matrix = false;
    for (int i = 0; i < 2; i++) {
      if (A("f", col_keys[i]).impl().type() == linear_map::SCALAR_MATRIX) {
        y_ = col_keys[i];
        x_ = col_keys[1-i];
        has_scalar_matrix = true;
        break;
      }
    }

    if (!has_scalar_matrix)
      return false;

    linear_map::LinearMap S = A("f", y_);
    alpha_ = static_cast<const linear_map::ScalarMatrixImpl&>(S.impl()).alpha();
    B_ = A("f", x_);

    // This case can just be handled the normal way
    if (B_.impl().m() <= B_.impl().n())
      return false;

    VLOG(2) << "Using graph form optimization, alpha=" << alpha_ << "\n"
            << "B: " << B_.impl().DebugString();

    BT_ = B_.Transpose();
    BTB_inv_ = (BT_*B_ + alpha_*alpha_*linear_map::Identity(
        B_.impl().n())).Inverse();

    if (b.has_key("f")) {
      c_ = b("f");
    } else {
      c_ = Eigen::VectorXd::Zero(B_.impl().m());
    }
    return true;
  }

  // Graph form
  bool graph_form_;
  std::string x_, y_;
  linear_map::LinearMap B_, BT_, BTB_inv_;
  Eigen::VectorXd c_;
  double alpha_;

  // Standard form
  BlockMatrix AT_, AAT_inv_, A_;
  BlockVector q_;
};
REGISTER_BLOCK_PROX_OPERATOR(LinearEqualityProx);
