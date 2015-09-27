#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/prox/newton.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

class NegativeLog {
public:
  static double f(const Eigen::VectorXd &x) {
    int n = x.rows();
    double sum = 0;
    for(int i=0; i<n; i++)
      sum += std::log(x(i));
    return -sum;
  }
  static Eigen::VectorXd gradf(const Eigen::VectorXd &x) {
    int n = x.rows();
    Eigen::VectorXd g(n);
    for(int i=0; i<n; i++)
      g(i) = -1 / (x(i));
    return g;
  }
  static Eigen::VectorXd hessf(const Eigen::VectorXd &x) {
    int n = x.rows();
    Eigen::VectorXd h(n);
    for(int i=0; i<n; i++)
      h(i) = 1 / (x(i)*x(i));
    return h;
  }
};

// \sum_i log(xi)
class NegativeLogProx final : public NewtonProx{
  double f(const Eigen::VectorXd &x) override {
    return NegativeLog::f(x);
  }
  Eigen::VectorXd gradf(const Eigen::VectorXd &x) override {
    return NegativeLog::gradf(x);
  }
  Eigen::VectorXd hessf(const Eigen::VectorXd &x) override {
    return NegativeLog::hessf(x);
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
REGISTER_PROX_OPERATOR(NegativeLogProx);

// I(-\sum_i log(xi) <= t)
class NegativeLogEpigraph final : public NewtonEpigraph{
  double f(const Eigen::VectorXd &x) override {
    return NegativeLog::f(x);
  }
  Eigen::VectorXd gradf(const Eigen::VectorXd &x) override {
    return NegativeLog::gradf(x);
  }
  Eigen::VectorXd hessf(const Eigen::VectorXd &x) override {
    return NegativeLog::hessf(x);
  }
  Eigen::VectorXd Apply(const Eigen::VectorXd& sv) override {
    return EpiByNewton(sv);
  }
};
REGISTER_PROX_OPERATOR(NegativeLogEpigraph);
