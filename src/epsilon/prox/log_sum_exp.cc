#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/prox/newton.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

// f(x) = log(\sum_i exp(x_i))
// f'(x) = exp(x_i) / (\sum_i exp(x_i)) = w
// f''(x) = diag(w) - exp(x_i) / (\sum_i exp(x_i))^2 exp(x_j)
//        = diag(w) - ww'
//
// H^{-1} = (I+lam*(diag(w) - ww'))^{-1}
//        = diag(1+lam w)^{-1} 
//        + lam * wi/(1+lam*wi) wj/(1+lam*wj) / (1-lam * \sum_i wi^2/(1+lam wi))
// H^{-1}g = gi/(1+lam wi) + wi/(1+lam wi) * lam*\sum_j wj/(1+lam wj)*gj / (...)

// H(x)^{-1} g(x),

class LogSumExp final : public SmoothFunction {
public:
  double eval(const Eigen::VectorXd &x) const override {
    int n = x.rows();
    double max_x = x.maxCoeff();
    Eigen::VectorXd w(n);
    for(int i=0; i<n; i++) {
      w(i) = std::exp(x(i)-max_x);
    }
    return max_x + std::log(w.sum());
  }
  Eigen::VectorXd gradf(const Eigen::VectorXd &x) const override {
    int n = x.rows();
    double max_x = x.maxCoeff();
    Eigen::VectorXd w(n);
    for(int i=0; i<n; i++) {
      w(i) = std::exp(x(i)-max_x);
    }
    w /= w.sum();
    return w;
  }
  Eigen::VectorXd hess_inv(
      const Eigen::VectorXd& lambda_,
      const Eigen::VectorXd& x,
      const Eigen::VectorXd& v) const override {
    const int n = x.rows();
    double lambda = lambda_(0);

    Eigen::VectorXd w = gradf(x);

    double t=0, r=0;
    for(int i=0; i<n; i++) {
      double wi = w(i);
      double vi = v(i);
      t += wi*wi/(1+lambda*wi);
      r += vi*wi/(1+lambda*wi);
    }
    double s = lambda*r/(1-lambda*t);
    Eigen::VectorXd Hinv_v(n);
    for(int i=0; i<n; i++) {
      double wi = w(i);
      double vi = v(i);
      Hinv_v(i) = vi/(1+lambda*wi) + wi/(1+lambda*wi) * s;
    }
    return Hinv_v;
  }
};

class LogSumExpProx : public NewtonProx {
public:
  LogSumExpProx() : NewtonProx(std::make_unique<LogSumExp>()) {}
};
REGISTER_PROX_OPERATOR(LOG_SUM_EXP, LogSumExpProx);

class LogSumExpEpigraph : public NewtonEpigraph {
public:
  LogSumExpEpigraph() : NewtonEpigraph(std::make_unique<LogSumExp>()) {}
};
REGISTER_EPIGRAPH_OPERATOR(LOG_SUM_EXP, LogSumExpEpigraph);
