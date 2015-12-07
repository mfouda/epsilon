
#include <float.h>

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/prox/newton.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

class NegativeEntropy final : public SmoothFunction {
public:
  double eval(const Eigen::VectorXd &x) const override {
    int n = x.rows();
    double sum = 0;
    for(int i=0; i<n; i++){
      if(x(i) <= 0)
        continue;
      sum += x(i)*std::log(x(i));
    }
    return sum;
  }
  Eigen::VectorXd gradf(const Eigen::VectorXd &x) const override {
    int n = x.rows();
    Eigen::VectorXd g(n);
    for(int i=0; i<n; i++)
      g(i) = 1+log(x(i));
    return g;
  }
  Eigen::VectorXd hessf(const Eigen::VectorXd &x) const override {
    int n = x.rows();
    Eigen::VectorXd h(n);
    for(int i=0; i<n; i++)
      h(i) = 1/x(i);
    return h;
  }

  Eigen::VectorXd proj_feasible(const Eigen::VectorXd& x) const override {
    return x.cwiseMax(1e-6);
  }
};

// \sum_i xi log xi
class SumNegEntrProx final : public NewtonProx {
public:
  SumNegEntrProx() : NewtonProx(std::make_unique<NegativeEntropy>()) {}
};
REGISTER_PROX_OPERATOR(SUM_NEG_ENTR, SumNegEntrProx);

// I(\sum_i xi log xi <= t)
class SumNegEntrEpigraph final : public ImplicitNewtonEpigraph {
public:
  SumNegEntrEpigraph()
    : ImplicitNewtonEpigraph(std::make_unique<NegativeEntropy>()) {}
};
REGISTER_EPIGRAPH_OPERATOR(SUM_NEG_ENTR, SumNegEntrEpigraph);
