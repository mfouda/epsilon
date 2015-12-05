
#include <float.h>

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/vector_prox.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

// \sum_i log(xi)
class SumNegLogProx final : public VectorProx {
 protected:
  virtual void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override {
    const Eigen::VectorXd& v = input.value_vec(0);
    const Eigen::VectorXd& lambda = input.lambda_vec();
    const int n = v.rows();
    VectorXd x(n);

    for(int i=0; i<n; i++){
      if(v(i) >= 0)
        x(i) = (v(i) + std::sqrt(v(i)*v(i) + 4*lambda(i)))/2;
      else
        x(i) = 2*lambda(i) / (-v(i) + std::sqrt(v(i)*v(i) + 4*lambda(i)));
    }

    output->set_value(0, x);
  }
};

REGISTER_PROX_OPERATOR(SUM_NEG_LOG, SumNegLogProx);
