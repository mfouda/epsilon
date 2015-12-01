
#include <float.h>

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"
#include "epsilon/prox/elementwise.h"
#include <cmath>

// \sum_i log(xi)
class SumNegLogProx : public ElementwiseProx {
 protected:
  Eigen::VectorXd ApplyElementwise(
      const Eigen::VectorXd& lambda,
      const Eigen::VectorXd& v) {
    int n = v.rows();
    VectorXd x(n);

    for(int i=0; i<n; i++){
      if(v(i) >= 0)
        x(i) = (v(i) + std::sqrt(v(i)*v(i) + 4*lambda(i)))/2;
      else
        x(i) = 2*lambda(i) / (-v(i) + std::sqrt(v(i)*v(i) + 4*lambda(i)));
    }
    return x;
  }
};

REGISTER_PROX_OPERATOR(SUM_NEG_LOG, SumNegLogProx);
