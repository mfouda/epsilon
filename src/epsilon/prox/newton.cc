#include "epsilon/prox/newton.h"

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"

Eigen::VectorXd NewtonProx::residual(
    const Eigen::VectorXd &x, const Eigen::VectorXd &v, double lam) {
  return x-v+lam*f_->gradf(x);
}

Eigen::VectorXd NewtonProx::Apply(const Eigen::VectorXd &v) {
  int n = v.rows();
  double eps = std::max(1e-12, 1e-10/n);

  // init
  Eigen::VectorXd x = f_->proj_feasible(v);

  int iter = 0;
  int MAX_ITER = 100;
  for(; iter < MAX_ITER; iter++) {
    Eigen::VectorXd hx = Eigen::VectorXd::Constant(n, 1.) + lambda_ * f_->hessf(x);
    Eigen::VectorXd gx = residual(x, v, lambda_);
    Eigen::VectorXd dx = (gx.array() / hx.array()).matrix();
//    VLOG(2) << "Iter " << iter << " gx: " << VectorDebugString(gx);

    // line search
    double beta = 0.001;
    double gamma = 0.5;
    double theta = 1;
    double x_res = gx.norm();
    while(theta > eps) {
      Eigen::VectorXd nx = x - theta * dx;
      double nx_res = residual(nx, v, lambda_).norm();
      if(nx_res <= (1-beta*theta)*x_res) {
        x = nx;
        x_res = nx_res;
        break;
      }
      theta *= gamma;
    }

    if(x_res < eps) {
      VLOG(1) << "Using " << iter+1 << " Newton iteration.\n";
      break;
    } else if(iter == MAX_ITER-1) {
      VLOG(1) << "Newton Method won't converge for prox, lam = " << lambda_ << ", xres = " << x_res << "\n";
    }
  }
  return x;
}

Eigen::VectorXd NewtonEpigraph::residual
(const Eigen::VectorXd &x, double t, double lam, const Eigen::VectorXd &v, double s) {
  int n = x.rows();
  VectorXd r(x.rows()+2);
  r.head(n) = x-v+lam*f_->gradf(x);
  r(n) = t-s-lam;
  r(n+1) = f_->eval(x)-t;
  return r;
}

Eigen::VectorXd NewtonEpigraph::Apply(const Eigen::VectorXd &sv) {
  int n = sv.rows()-1;
  double eps = std::max(1e-12, 1e-10/n);

  double s = sv(0);
  Eigen::VectorXd v = sv.tail(n);
  Eigen::VectorXd x = f_->proj_feasible(v);
  double feasible_dist = (v-x).norm();

  // easy case
  if (feasible_dist < eps && f_->eval(x) <= s)
    return sv;

  // init
  double t = s;
  double lam = 1;

  int iter = 0;
  int MAX_ITER = 100;
  for(; iter < MAX_ITER; iter++) {
    Eigen::VectorXd hx = Eigen::VectorXd::Constant(n, 1.) + lam * f_->hessf(x);
    Eigen::VectorXd g = residual(x, t, lam, v, s);
    VLOG(2) << "Iter " << iter << "\n"
            << " lam: " << lam << "\n"
            << " x: " << VectorDebugString(x) << "\n"
            << " g: " << VectorDebugString(g) << "\n"
            << " hx: " << VectorDebugString(hx);

    // construct arrowhead hessian matrix
    Eigen::VectorXd d(n+1);
    d.head(n) = Eigen::VectorXd::Constant(n, 1.) + lam * f_->hessf(x);
    d(n) = 1;
    Eigen::VectorXd z(n+1);
    z.head(n) = g.head(n);
    z(n) = -1.;
    double alpha = 0;
    Eigen::VectorXd step = SolveArrowheadSystem(d, z, alpha, g);
    VLOG(2) << " step: " << VectorDebugString(step);

    // line search
    double beta = 0.001;
    double gamma = 0.5;
    double theta = 1;
    double x_res = g.norm();
    while(theta > eps) {
      Eigen::VectorXd nx = x - theta*step.head(n);
      nx = f_->proj_feasible(nx);
      double nt = t - theta*step(n);
      double nlam = lam - theta*step(n+1);
      if(nlam < eps)
              nlam = eps;
      double nx_res = residual(nx, nt, nlam, v, s).norm();
      VLOG(2) << "x_res = " << x_res << ", nx_res = " << nx_res << "\n";
      if(nx_res <= (1-beta*theta)*x_res) {
        x = nx;
        t = nt;
        lam = nlam;
        x_res = nx_res;
        break;
      }
      theta *= gamma;
      if(theta < eps)
              VLOG(1) << "Line search reach max iter, x_res=" << x_res << "\n";
    }
    VLOG(2) << "XRES = "<< x_res << ", theta = " << theta <<"\n";

    if(x_res < eps) {
      VLOG(1) << "Using " << iter+1 << " Newton iteration.\n";
      break;
    } else if(iter == MAX_ITER-1) {
      VLOG(1) << "Newton Method won't converge for epigraph.\n"
        << "sv = " << sv << '\n';
    }
  }
  Eigen::VectorXd tx(n+1);
  tx(0) = t;
  tx.tail(n) = x;
  return tx;
}

// solve Ax=b, where A = [diag(d), z; z', alpha], and d_i>0 forall i
Eigen::VectorXd NewtonEpigraph::SolveArrowheadSystem
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

Eigen::VectorXd ImplicitNewtonEpigraph::Apply(const Eigen::VectorXd& sv) {
  int n = sv.rows()-1;
  double s = sv(0);
  Eigen::VectorXd v = sv.tail(n);
  double t = s;
  Eigen::VectorXd x = f_->proj_feasible(v);

  if(f_->eval(x) <= t) {
    Eigen::VectorXd txx(n+1);
    txx(0) = s;
    txx.tail(n) = x;
    return txx;
  }

  double lam = 1;
  int iter = 0, max_iter=100;
  double res = 0;
  for(; iter < max_iter; iter++) {
    //ProxOperatorArg prox_arg(lam, NULL, NULL);
    //NewtonProx::Init(prox_arg);
    lambda_ = lam;
    x = NewtonProx::Apply(v);

    Eigen::VectorXd gx = f_->gradf(x);
    Eigen::VectorXd hx = f_->hessf(x);
    double glam = f_->eval(x) - lam - s;
    double hlam = -(gx.cwiseQuotient((1.+lam*hx.array()).matrix()).dot(gx)) - 1;
    VLOG(2) << "glam = " << glam << ", hlam = " << hlam << "\n";

    res = std::abs(glam);
    if(res < 1e-10)
      break;

    lam = lam - glam/hlam;
    if (lam < 0)
      lam = 1e-6;

  }
  if(iter == max_iter) {
    VLOG(2) << "Newton reach max iter, residual = " << res << "\n";
  } else {
    VLOG(2) << "Newton ends in " << iter << "iterations, r = " << res << "\n";
  }

  Eigen::VectorXd tx(n+1);
  tx(0) = s+lam;
  tx.tail(n) = NewtonProx::Apply(v);

  return tx;
}
