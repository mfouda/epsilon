#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/vector_prox.h"
#include "epsilon/vector/vector_util.h"

void ApplyKLDivProx(
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v,
    Eigen::VectorXd& x,
    Eigen::VectorXd& y) {
  const double eps = 1e-13;
  const int n = u.rows();

  for(int i=0; i<n; i++) {
    int iter = 0, max_iter=1000;
    double res = 0;
    double yhat = std::max((0.5+lambda(i)-v(i))/lambda(i), eps);
    if(std::abs(u(i)) < eps*eps and std::abs(v(i)) < eps*eps){
      x(i) = u(i);
      y(i) = v(i);
      continue;
    }
    // Solve:
    //   q(lam-v+q)/lam-u+lam*log(lam-v+q)/lam = 0
    // Let yhat = (lam-v+q)/lam, then
    //   lam*yhat*yhat-(1-v/lam)yhat-u+lam*log(yhat)=0
    for(; iter < max_iter; iter++) {
      double f = lambda(i)*yhat*yhat + (v(i)-lambda(i))*yhat - u(i)
                 + lambda(i)*std::log(yhat);
      double F = 2*lambda(i)*yhat + (v(i)-lambda(i)) + lambda(i)/yhat;
      res = f/F;
      if(std::abs(res) < eps ||
         (yhat <= eps*2 && f/F > 0) ||
         (lambda(i)*yhat+v(i)-lambda(i) <= eps*2 && f/F > 0)) {
        break;
      }

      yhat = yhat - f/F;
      if(yhat < eps)
        yhat = eps;
      if(lambda(i)*yhat+v(i)-lambda(i) < eps)
        yhat = (eps+lambda(i)-v(i))/lambda(i);
    }
    if(iter == max_iter)
      VLOG(2) << "Newton does not converge for kl_div prox\n";
    VLOG(2) << "newton_iter = " << iter << ", f = " << res << "\n";
    y(i) = lambda(i)*yhat + v(i)-lambda(i);
    x(i) = y(i)*yhat;
  }
}

class SumKLDivProx final : public VectorProx {
protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override {
    const int n = input.value_vec(0).rows();
    Eigen::VectorXd x(n), y(n);
    ApplyKLDivProx(
        input.lambda_vec(), input.value_vec(0), input.value_vec(1), x, y);
    output->set_value(0, x);
    output->set_value(1, y);
  }
};
REGISTER_PROX_OPERATOR(SUM_KL_DIV, SumKLDivProx);

class SumKLDivEpigraph final : public VectorProx {
protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override {
    const Eigen::VectorXd& u = input.value_vec(0);
    const Eigen::VectorXd& v = input.value_vec(1);
    const double s = input.value(2);

    const double eps = 1e-10;
    const int n = u.rows();

    Eigen::VectorXd x(n), y(n);
    int iter = 0, max_iter = 100;
    double res = 0;
    double lambda = 1;
    for(; iter<max_iter; iter++) {
      Eigen::VectorXd g(2);
      Eigen::MatrixXd h(2,2);

      ApplyKLDivProx(Eigen::VectorXd::Constant(n, lambda), u, v, x, y);
      double glam = -s - lambda;
      double hlam = -1;
      for(int i=0; i<n; i++) {
        glam += x(i)*std::log(x(i)/y(i)) - x(i) + y(i);
        g(0) = std::log(x(i)/y(i));
        g(1) = -x(i)/y(i)+1;
        h(0,0) = 1/x(i);
        h(0,1) = -1/y(i);
        h(1,0) = h(0,1);
        h(1,1) = x(i)/(y(i)*y(i));
        h = Eigen::MatrixXd::Identity(2, 2) + lambda * h;

        hlam -= (g.transpose() * h.inverse() * g);
      }
      res = glam;
      if(std::abs(res) < eps or (lambda<=eps*2 and glam/hlam > 0))
        break;
      lambda = lambda - glam/hlam;
      if(lambda < eps)
        lambda = eps;
    }
    if(iter == max_iter)
      VLOG(2) << "Newton method won't converge for implicit kl_div\n";
    VLOG(2) << "Use " << iter << " Newton Iters, res = " << res << "\n";

    ApplyKLDivProx(Eigen::VectorXd::Constant(n, lambda), u, v, x, y);
    output->set_value(0, x);
    output->set_value(1, y);
    output->set_value(2, s + lambda);
  }
};
REGISTER_EPIGRAPH_OPERATOR(SUM_KL_DIV, SumKLDivEpigraph);
