#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

Eigen::VectorXd GetElementwiseMultiply(
    const Expression& expr,
    const VariableOffsetMap& var_map) {
  SparseXd A = GetSparseAffineOperator(expr, var_map);
  CHECK(IsDiagonal(A));
  return A.diagonal();
}

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
REGISTER_PROX_OPERATOR(NormL1Prox);

// I(||x||_1 <= t)
class NormL1Epigraph final : public ProxOperator {
  Eigen::VectorXd Apply(const Eigen::VectorXd& sv) override {
    const int n = sv.rows() - 1;
    const double s = sv(0);
    Eigen::VectorXd a = sv.tail(n).cwiseAbs();
    const double v_norm_l1 = a.sum();
    const double v_norm_inf = a.maxCoeff();

    if (v_norm_inf <= -s) {
      return Eigen::VectorXd::Zero(sv.rows());
    } else if (v_norm_l1 <= s) {
      return sv;
    } else {
      sort<double>(a);
      a.reverseInPlace();
      int i = 0;
      double acc = -s;
      for(; i<n; i++){
        double lam = acc/(i+1);
        if(a(i) <= lam)
          break;
        acc = acc + a(i);
      }
      double lam = acc/(i+1);
      Eigen::VectorXd v = softThres(sv, lam);
      v(0) = s+lam;
      return v;
    }
  }

  template <typename ScalarType, typename Derived>
  void sort(Eigen::MatrixBase<Derived> &xValues)
  {
    std::sort(xValues.derived().data(), xValues.derived().data()+xValues.derived().size());
  }

  Eigen::VectorXd softThres(const Eigen::VectorXd &v, double lam){
    return (v.array()-Eigen::ArrayXd::Constant(v.rows(), lam)).max(0)
      - (-v.array()-Eigen::ArrayXd::Constant(v.rows(), lam)).max(0);
  }
};
REGISTER_PROX_OPERATOR(NormL1Epigraph);
