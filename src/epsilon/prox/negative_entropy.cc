
#include <float.h>

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/prox/newton.h"
#include "epsilon/vector/dynamic_matrix.h"
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
    return x.cwiseMax(1e-8);
  }
};

// \sum_i xi log xi
class NegativeEntropyProx final : public NewtonProx {
public:
  NegativeEntropyProx() : NewtonProx(std::make_unique<NegativeEntropy>()) {}
};
REGISTER_PROX_OPERATOR(NegativeEntropyProx);

// I(\sum_i xi log xi <= t)
class NegativeEntropyEpigraph final : public NewtonEpigraph{
public:
  NegativeEntropyEpigraph() : NewtonEpigraph(std::make_unique<NegativeEntropy>()) {}
};
REGISTER_PROX_OPERATOR(NegativeEntropyEpigraph);
