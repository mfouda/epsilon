#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/prox/newton.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

class Logistic final : public SmoothFunction {
public:
  double eval(const Eigen::VectorXd &x) const override {
    int n = x.rows();
    double sum = 0;
    for(int i=0; i<n; i++){
      sum += std::log(1+std::exp(x(i)));
    }
    return sum;
  }
  Eigen::VectorXd gradf(const Eigen::VectorXd &x) const override {
    int n = x.rows();
    Eigen::VectorXd g(n);
    for(int i=0; i<n; i++)
      g(i) = std::exp(x(i)) / (1+std::exp(x(i)));
    return g;
  }
  Eigen::VectorXd hessf(const Eigen::VectorXd &x) const override {
    int n = x.rows();
    Eigen::VectorXd h(n);
    for(int i=0; i<n; i++)
      h(i) = std::exp(x(i)) / std::pow(1+std::exp(x(i)), 2);
    return h;
  }
};

class SumLogisticProx final : public NewtonProx {
public:
  SumLogisticProx() : NewtonProx(std::make_unique<Logistic>()) {}
};
REGISTER_PROX_OPERATOR(SUM_LOGISTIC, SumLogisticProx);

// class LogisticEpigraph final : public NewtonEpigraph {
// public:
//   LogisticEpigraph() : NewtonEpigraph(std::make_unique<Logistic>()) {}
// };
// REGISTER_PROX_OPERATOR(LogisticEpigraph);
