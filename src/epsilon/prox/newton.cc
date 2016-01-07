#include "epsilon/prox/newton.h"

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

Eigen::VectorXd ProxResidual(
    const SmoothFunction& f,
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& v) {
  return x-v+lambda.asDiagonal()*f.gradf(x);
}

Eigen::VectorXd EpigraphResidual(
    const SmoothFunction& f,
    double lam,
    const Eigen::VectorXd& x, double t,
    const Eigen::VectorXd& v, double s) {
  int n = x.rows();
  VectorXd r(x.rows()+2);
  r.head(n) = x-v+lam*f.gradf(x);
  r(n) = t-s-lam;
  r(n+1) = f.eval(x)-t;
  return r;
}

// solve Ax=b, where A = [diag(d), z; z', alpha], and d_i>0 forall i
Eigen::VectorXd SolveArrowheadSystem(
    const Eigen::VectorXd &d,
    const Eigen::VectorXd &z,
    double alpha,
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

Eigen::VectorXd ApplyNewtonProx(
    const SmoothFunction& f,
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& v) {
  int n = v.rows();
  double eps = std::max(1e-12, 1e-10/n);

  // init
  Eigen::VectorXd x = f.proj_feasible(v);

  int iter = 0;
  int MAX_ITER = 100;
  for(; iter < MAX_ITER; iter++) {
    Eigen::VectorXd gx = ProxResidual(f, lambda, x, v);
    Eigen::VectorXd dx = f.hess_inv(lambda, x, gx);
    VLOG(3) << "Iter " << iter << " gx: " << VectorDebugString(gx);

    // line search
    double beta = 0.001;
    double gamma = 0.5;
    double theta = 1;
    double x_res = gx.norm();
    while(theta > eps) {
      Eigen::VectorXd nx = x - theta * dx;
      double nx_res = ProxResidual(f, lambda, nx, v).norm();
      if(nx_res <= (1-beta*theta)*x_res) {
        x = nx;
        x_res = nx_res;
        break;
      }
      theta *= gamma;
    }

    if(x_res < eps) {
      VLOG(2) << "Using " << iter+1 << " Newton iteration.\n";
      break;
    } else if(iter == MAX_ITER-1) {
      VLOG(2) << "Newton Method won't converge for prox, lam = "
              << VectorDebugString(lambda)
              << ", xres = " << x_res << "\n";
    }
  }
  return x;
}

Eigen::VectorXd ApplyNewtonProx(
    const SmoothFunction& f,
    double lambda,
    const Eigen::VectorXd& v) {
  return ApplyNewtonProx(
      f, Eigen::VectorXd::Constant(v.rows(), lambda), v);
}

void NewtonProx::ApplyVector(
    const VectorProxInput& input,
    VectorProxOutput* output) {
  output->set_value(
      0, ApplyNewtonProx(
          *f_, input.lambda_vec(), input.value_vec(0)));
}

void NewtonEpigraph::ApplyVector(
    const VectorProxInput& input,
    VectorProxOutput* output) {
  const SmoothFunction& f = *f_;
  const Eigen::VectorXd& v = input.value_vec(0);
  const double s =  input.value(1);
  const int n = v.rows();
  const double eps = std::max(1e-12, 1e-10/n);

  Eigen::VectorXd x = f.proj_feasible(v);
  double feasible_dist = (v-x).norm();

  // easy case
  if (feasible_dist < eps && f.eval(x) <= s) {
    output->set_value(0, v);
    output->set_value(1, s);
    return;
  }

  // init
  double t = s;
  double lam = 1;

  int iter = 0;
  int MAX_ITER = 100;
  for(; iter < MAX_ITER; iter++) {
    Eigen::VectorXd g = EpigraphResidual(f, lam, x, t, v, s);
    VLOG(2) << "Iter " << iter << "\n"
            << " lam: " << lam << "\n"
            << " x: " << VectorDebugString(x) << "\n"
            << " g: " << VectorDebugString(g);

    // construct arrowhead hessian matrix from newton step
    Eigen::VectorXd nt_step = f.hess_inv(lam, x, g.head(n));
    double nt_res = g.head(n).dot(nt_step);
    Eigen::VectorXd step(n+2);
    double scale = (-nt_res+g(n)+g(n+1))/(nt_res+1);
    step.head(n) = (1+scale)*nt_step;
    step(n) = g(n);
    step(n+1) = 0;
    step(n) -= scale;
    step(n+1) -= scale;
    VLOG(2) << " step: " << VectorDebugString(step);

    // line search
    double beta = 0.001;
    double gamma = 0.5;
    double theta = 1;
    double x_res = g.norm();
    while(theta > eps) {
      Eigen::VectorXd nx = x - theta*step.head(n);
      nx = f.proj_feasible(nx);
      double nt = t - theta*step(n);
      double nlam = lam - theta*step(n+1);
      if(nlam < eps)
              nlam = eps;
      double nx_res = EpigraphResidual(f, nlam, nx, nt, v, s).norm();
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
      VLOG(1) << "Newton Method won't converge for epigraph.\n";
    }
  }

  output->set_value(0, x);
  output->set_value(1, t);
}

void ImplicitNewtonEpigraph::ApplyVector(
    const VectorProxInput& input,
    VectorProxOutput* output) {

  const SmoothFunction& f = *f_;
  const Eigen::VectorXd& v = input.value_vec(0);
  const double s =  input.value(1);

  Eigen::VectorXd x = f.proj_feasible(v);
  double t = s;

  if (f.eval(x) <= t) {
    output->set_value(0, x);
    output->set_value(1, t);
    return;
  }

  double lam = 1;
  int iter = 0, max_iter=100;
  double res = 0;
  for(; iter < max_iter; iter++) {
    x = ApplyNewtonProx(*f_, lam, v);

    Eigen::VectorXd gx = f.gradf(x);
    double glam = f.eval(x) - lam - s;
    double hlam = -(f.hess_inv(lam, x, gx).dot(gx)) - 1;
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

  output->set_value(0, ApplyNewtonProx(*f_, lam, v));
  output->set_value(1, s+lam);
}

void BisectionEpigraph::ApplyVector(
    const VectorProxInput& input_,
    VectorProxOutput* output) {

  const double s =  input_.value(1);

  output->set_value(0, input_.value_vec(0));
  double fval = prox_->Eval(output);
  if (fval <= s) {
    VLOG(2) << "Bisection: Easy case in epigraph.\n";
    output->set_value(1, s);
    return;
  } 

  double lam = 1;
  double eps = 1e-5;
  int iter = 0, max_iter=100;
  double upper = lam, lower = 0;
  bool upper_fixed = false;

  VectorProxInput input(input_);
  input.set_lambda(lam);

  for(; iter<max_iter; iter++) {
    prox_->ApplyVector(input, output);
    double fval = prox_->Eval(output);

    double g = fval-(lam+s); 
    VLOG(2) << "Bisection: g = " << g 
      << " lambda = " << lam 
      << " in (" << lower << ", " << upper << ")\n";
    if (fabs(g) <= eps) {
      output->set_value(1, lam + s);
      return;
    } else {
      if(g > 0 and not upper_fixed) {
        lam *= 2;
        upper = lam;
      } else if (g > 0) {
        lower = lam;
        lam = (lam+upper)/2;
      } else {
        upper = lam;
        lam = (lam+lower)/2;
        upper_fixed = true;
      }
    }
    input.set_lambda(lam);
  }
}
