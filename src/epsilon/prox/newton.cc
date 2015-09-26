#include "epsilon/prox/newton.h"

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

Eigen::VectorXd NewtonProx::residual(const Eigen::VectorXd &x, const Eigen::VectorXd &v, double lam) {
  return x-v+lam*gradf(x);
}

Eigen::VectorXd NewtonProx::ProxByNewton(const Eigen::VectorXd &v, double lam) {
  int n = v.rows();
  double eps = std::max(1e-12, 1e-10/n);

  // init
  Eigen::VectorXd x = v;

  int iter = 0;
  int MAX_ITER = 100;
  for(; iter < MAX_ITER; iter++) {
    Eigen::VectorXd hx = Eigen::VectorXd::Constant(n, 1.) + lam * hessf(x);
    Eigen::VectorXd gx = residual(x, v, lam);
    
    Eigen::VectorXd dx = (gx.array() / hx.array()).matrix();

    // line search
    double beta = 0.001;
    double gamma = 0.5;
    double theta = 1;
    double x_res = gx.norm();
    while(theta > eps) {
      Eigen::VectorXd nx = x - theta * dx;
      double nx_res = residual(nx, v, lam).norm();
      if(nx_res <= (1-beta*theta)*x_res) {
        x = nx;
        x_res = nx_res; 
        break;
      }
      theta *= gamma;
    }

    if(x_res < eps){
      break;
    }
  }
  return x;
}

Eigen::VectorXd NewtonEpigraph::residual
(const Eigen::VectorXd &x, double t, double lam, const Eigen::VectorXd &v, double s) {
  int n = x.rows();
  VectorXd r(x.rows()+2);
  r.head(n) = x-v+lam*gradf(x);
  r(n) = t-s-lam;
  r(n+1) = f(x)-t;
  return r;
}

Eigen::VectorXd NewtonEpigraph::EpiByNewton(const Eigen::VectorXd &v, double s) {
  int n = v.rows();
  double eps = std::max(1e-12, 1e-10/n);
  
  // init
  Eigen::VectorXd x = v;
  double t = s;
  double lam = 1;

  int iter = 0;
  int MAX_ITER = 100;
  for(; iter < MAX_ITER; iter++) {
    Eigen::VectorXd hx = Eigen::VectorXd::Constant(n, 1.) + lam * hessf(x);
    Eigen::VectorXd g = residual(x, t, lam, v, s);

    // construct arrowhead hessian matrix
    Eigen::VectorXd d(n+1);
    d.head(n) = Eigen::VectorXd::Constant(n, 1.) + lam * hessf(x);
    d(n) = 1;
    Eigen::VectorXd z(n+1);
    z.head(n) = g.head(n);
    z(n) = -1.;
    double alpha = 0;
    Eigen::VectorXd step = solve_arrowhead_system(d, z, alpha, g);

    // line search
    double beta = 0.001;
    double gamma = 0.5;
    double theta = 1;
    double x_res = g.norm();
    while(theta > eps) {
      Eigen::VectorXd nx = x - theta*step.head(n);
      double nt = t - theta*step(n);
      double nlam = lam - theta*step(n+1);
      double nx_res = residual(nx, nt, nlam, v, s).norm();
      if(nx_res <= (1-beta*theta)*nx_res) {
        x = nx;
        t = nt;
        lam = nlam;
        x_res = nx_res;
        break;
      }
      theta *= gamma;
    }

    if(x_res < eps)
      break;
  }
  return x;
}

// solve Ax=b, where A = [diag(d), z; z', alpha], and d_i>0 forall i
Eigen::VectorXd NewtonEpigraph::solve_arrowhead_system
  (const Eigen::VectorXd &d, const Eigen::VectorXd &z, double alpha,
    const Eigen::VectorXd &b) {
    int n = z.rows();
    Eigen::VectorXd u = z.array() / d.array();
    double rho = alpha - u.dot(z);
    Eigen::VectorXd y(n+1);
    y.head(n) = u;
    y(n) = -1;
    Eigen::VectorXd dinv_b(n+1);
    dinv_b.head(n) = b.head(n).array() / d.array();
    dinv_b(n) = 0;

    return dinv_b + 1/rho * y.dot(b) * y;
}
