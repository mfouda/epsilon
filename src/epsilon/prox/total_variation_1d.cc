
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/vector.h"
#include "epsilon/vector/vector_util.h"

extern "C" {
void tf_dp (int n, double *y, double lam, double *beta);
}

// lam*||x[2:] - x[:-1]||_1
class TotalVariation1DProx final : public VectorProx {
protected:
  Eigen::VectorXd ApplyVector(double lambda, const Eigen::VectorXd& v) override {
    const int n = v.rows();
    Eigen::VectorXd x(n);
    tf_dp(n, const_cast<double*>(v.data()), lambda, x.data());
    return x;
  }
};
REGISTER_PROX_OPERATOR(TOTAL_VARIATION_1D, TotalVariation1DProx);
