
#include <float.h>

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/prox/newton.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

class NegativeLog final : public SmoothFunction {
public:
  double eval(const Eigen::VectorXd& x) const override {
    int n = x.rows();
    double sum = 0;
    for(int i=0; i<n; i++)
      sum += std::log(x(i));
    return -sum;
  }

  Eigen::VectorXd gradf(const Eigen::VectorXd& x) const override {
    int n = x.rows();
    Eigen::VectorXd g(n);
    for(int i=0; i<n; i++)
      g(i) = -1 / (x(i));
    return g;
  }

  Eigen::VectorXd hessf(const Eigen::VectorXd& x) const override {
    int n = x.rows();
    Eigen::VectorXd h(n);
    for(int i=0; i<n; i++)
      h(i) = 1 / (x(i)*x(i));
    return h;
  }

  Eigen::VectorXd proj_feasible(const Eigen::VectorXd& x) const override {
    return x.cwiseMax(DBL_MIN);
  }
};

// \sum_i log(xi)
class NegativeLogProx final : public NewtonProx {
public:
  NegativeLogProx() : NewtonProx(std::make_unique<NegativeLog>()) {}
};
REGISTER_PROX_OPERATOR(NegativeLogProx);

// I(-\sum_i log(xi) <= t)
class NegativeLogEpigraph final : public NewtonEpigraph {
public:
  NegativeLogEpigraph() : NewtonEpigraph(std::make_unique<NegativeLog>()) {}
};
REGISTER_PROX_OPERATOR(NegativeLogEpigraph);
