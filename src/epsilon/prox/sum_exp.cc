#include <float.h>

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/prox/newton.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

// \sum_i exp(x(i))
class SumExp final : public ElemwiseSmoothFunction {
public:
  double eval(const Eigen::VectorXd &x) const override {
    int n = x.rows();
    double sum = 0;
    for(int i=0; i<n; i++){
      sum +=  std::exp(x(i));
    }
    return sum;
  }
  Eigen::VectorXd gradf(const Eigen::VectorXd &x) const override {
    int n = x.rows();
    Eigen::VectorXd g(n);
    for(int i=0; i<n; i++)
      g(i) = std::exp(x(i));
    return g;
  }
  Eigen::VectorXd hessf(const Eigen::VectorXd &x) const override {
    int n = x.rows();
    Eigen::VectorXd h(n);
    for(int i=0; i<n; i++)
      h(i) = std::exp(x(i));
    return h;
  }
};

class SumExpProx : public NewtonProx {
public:
  SumExpProx() : NewtonProx(std::make_unique<SumExp>()) {}
};
REGISTER_PROX_OPERATOR(SUM_EXP, SumExpProx);

class SumExpEpigraph : public NewtonEpigraph {
public:
  SumExpEpigraph() : NewtonEpigraph(std::make_unique<SumExp>()) {}
};
REGISTER_EPIGRAPH_OPERATOR(SUM_EXP, SumExpEpigraph);
