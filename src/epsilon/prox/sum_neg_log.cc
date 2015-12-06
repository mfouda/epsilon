
#include <float.h>

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/vector_prox.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

Eigen::VectorXd ApplyNegLogProx(
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& v) {
  const int n = v.rows();
  Eigen::VectorXd x(n);

  for(int i=0; i<n; i++){
    if(v(i) >= 0)
      x(i) = (v(i) + std::sqrt(v(i)*v(i) + 4*lambda(i)))/2;
    else
      x(i) = 2*lambda(i) / (-v(i) + std::sqrt(v(i)*v(i) + 4*lambda(i)));
  }

  return x;
}

// \sum_i log(xi)
class SumNegLogProx final : public VectorProx {
 protected:
  virtual void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override {
    output->set_value(
        0, ApplyNegLogProx(
            input.lambda_vec(),
            input.value_vec(0)));
  }
};
REGISTER_PROX_OPERATOR(SUM_NEG_LOG, SumNegLogProx);

// I(-\sum_i log(xi) <= t)
class SumNegLogEpigraph final : public VectorProx {
public:
  virtual void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override {
    const Eigen::VectorXd& v = input.value_vec(0);
    const double s = input.value(1);
    const int n = v.rows();

    const double eps = 1e-10;
    double lambda = 1;

    int iter = 0;
    double g, h;
    for(; iter<1000; iter++) {
      g = -lambda -s;
      h = -1;
      for(int i=0; i<n; i++) {
        // Avoiding catastrophic cancellation
        double z = std::sqrt(v(i)*v(i) + 4*lambda);
        if(v(i) >= 0) { //
          g -= std::log((v(i) + z)/2);
          h -= 1. / (v(i)*(v(i)+z)/2 + 2*lambda);
        } else { // log x = -log 1/x
          g += std::log((-v(i) + z)/(2*lambda));
          h -= 1. / (v(i)*2*lambda/(-v(i)+z) + 2*lambda);
        }
      }

      if(std::fabs(g) <= eps)
        break;

      if(h >= -1e-10)
        h = -1e-10;

      lambda -= g/h;
      if(lambda <= 1e-10)
        lambda = 1e-10;
    }
    VLOG(2) << iter+1 << " Newton iteration used on negative log epigraph." << "\n";
    VLOG(2) << "g = " << g << "\n";
    VLOG(2) << "lambda = " << lambda << "\n";

    output->set_value(0, ApplyNegLogProx(
        Eigen::VectorXd::Constant(n, lambda), v));
    output->set_value(1, s + lambda);
  }
};
REGISTER_EPIGRAPH_OPERATOR(SUM_NEG_LOG, SumNegLogEpigraph);
