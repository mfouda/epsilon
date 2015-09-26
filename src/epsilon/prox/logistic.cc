#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/prox/newton.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

class Logistic {
public:
  static double f(const Eigen::VectorXd &x) {
    int n = x.rows();
    double sum = 0;
    for(int i=0; i<n; i++)
      sum += std::log(1+std::exp(x(i)));
    return sum;
  }
  static Eigen::VectorXd gradf(const Eigen::VectorXd &x) {
    int n = x.rows();
    Eigen::VectorXd g(n);
    for(int i=0; i<n; i++)
      g(i) = std::exp(x(i)) / (1+std::exp(x(i)));
    return g;
  }
  static Eigen::VectorXd hessf(const Eigen::VectorXd &x) {
    int n = x.rows();
    Eigen::VectorXd h(n);
    for(int i=0; i<n; i++)
      h(i) = std::exp(x(i)) / pow(1+std::exp(x(i)), 2);
    return h;
  }
};

// lam*||x||_2
class LogisticProx final : public NewtonProx{
  double f(const Eigen::VectorXd &x) override {
    return Logistic::f(x);
  }
  Eigen::VectorXd gradf(const Eigen::VectorXd &x) override {
    return Logistic::gradf(x);
  }
  Eigen::VectorXd hessf(const Eigen::VectorXd &x) override {
    return Logistic::hessf(x);
  }
  void Init(const ProxOperatorArg& arg) override {
    // Expression tree:
    // TODO
    //   VARIABLE (x)
    lambda_ = arg.lambda();
  }
  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    return ProxByNewton(v, lambda_);
  }

private:
  double lambda_;
};
REGISTER_PROX_OPERATOR(LogisticProx);

// I(||x||_2 <= t)
class LogisticEpigraph final : public NewtonEpigraph{
  double f(const Eigen::VectorXd &x) override {
    return Logistic::f(x);
  }
  Eigen::VectorXd gradf(const Eigen::VectorXd &x) override {
    return Logistic::gradf(x);
  }
  Eigen::VectorXd hessf(const Eigen::VectorXd &x) override {
    return Logistic::hessf(x);
  }
  Eigen::VectorXd Apply(const Eigen::VectorXd& sv) override {
    int n = sv.rows() - 1;
    return EpiByNewton(sv.tail(n), sv(0));
  }
};
REGISTER_PROX_OPERATOR(LogisticEpigraph);
