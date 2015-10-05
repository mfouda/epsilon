#include "epsilon/prox/scaled_zone.h"

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"
#include <cassert>

bool abs_cmp_descending(const double &x, const double &y)
{
    double ax = std::fabs(x), ay = std::fabs(y);
    return ax > ay;
}

double ScaledZoneProx::key(double x) {
  if(x>0)
    return (x-M_) / alpha_;
  else
    return (x+M_) / beta_;
}

Eigen::VectorXd ScaledZoneProx::Apply(const Eigen::VectorXd &v) {
  int n = v.rows();
  Eigen::VectorXd x = (v.array()-C_).matrix();

  for(int i=0; i<n; i++){ // possible loop unrolling
    if(std::fabs(x(i)) <= M_)
      x(i) = x(i);
    else if(x(i) > M_ + lambda_ * alpha_)
      x(i) = x(i) - lambda_ * alpha_;
    else if(x(i) < -M_ - lambda_ * beta_)
      x(i) = x(i) + lambda_ * beta_;
    else if(x(i) > 0)
      x(i) = M_;
    else
      x(i) = -M_;
  }
  return (x.array()+C_).matrix();
}
REGISTER_PROX_OPERATOR(ScaledZoneProx);

std::string ArrayDebugString(double *a, int n)
{
        std::string t = "";
        for(int i=0; i<n; i++)
                t += StringPrintf("%.4lf ", a[i]);
        return t;
}

Eigen::VectorXd ScaledZoneEpigraph::Apply(const Eigen::VectorXd& sv) {
  const int n = sv.rows() - 1;
  const double s = sv(0);
  Eigen::VectorXd vec_y = sv.tail(n);
  double *y = vec_y.data();
  int ny = n;

  // Filter and eval function
  double fval = 0;
  for(int i=0; i<ny; ) {
    y[i] = y[i] - C_;
    if(std::fabs(y[i]) <= M_ || (y[i]>0 && alpha_==0) || (y[i]<0 && beta_==0)) {
            assert(ny >= 1);
      std::swap(y[i], y[--ny]);
    } else {
      if(y[i] > M_)
        fval += alpha_ * (y[i]-M_);
      else if(y[i] < -M_)
        fval += beta_ * (-y[i]-M_);

      y[i] = key(y[i]);
      i++;
    }
  }

  if (fval <= s){
    return sv;
  }

  std::sort(y, y+ny, abs_cmp_descending);

  double div = 0;
  double acc = -s;

  for(int i=0; i<ny; i++) {
    double lam = acc/(div+1);
    if(std::fabs(y[i]) <= lam)
      break;

    if(y[i] > 0) {
      acc = acc + alpha_*alpha_ * y[i];
      div = div + alpha_*alpha_;
    } else if(y[i] < 0) {
      acc = acc + beta_*beta_ * (-y[i]);
      div = div + beta_*beta_;
    }
  }
  double lam = acc/(div+1);

  Eigen::VectorXd tx(n+1);
  this->lambda_ = lam;
  tx(0) = s+lam;
  tx.tail(n) = ScaledZoneProx::Apply(sv.tail(n));
  return tx;
}
REGISTER_PROX_OPERATOR(ScaledZoneEpigraph);
