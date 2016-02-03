#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/vector_prox.h"
#include "epsilon/vector/vector_util.h"

struct ScaledZoneParams {
  Eigen::VectorXd alpha, beta;
  double M, C;
};

// Assumption: alpha,beta,M>=0, C in R
// f(x) = \sum_i \alpha \max(0, (x_i-C)-M) + \beta \max(0, -(x_i-C)-M)
class ScaledZoneProx final : public VectorProx {
public:
  void Init(const ProxOperatorArg& arg) override;

protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override;

private:
  ScaledZoneParams params_;
};

Eigen::VectorXd Promote(const Eigen::VectorXd& x, int n) {
  if (x.rows() == n)
    return x;
  if (x.rows() == 1 and n != 1)
    return Eigen::VectorXd::Constant(n, x(0));
  LOG(FATAL) << x.rows() << " != " << n;
}

ScaledZoneParams GetParams(const ProxFunction& prox) {
  CHECK_EQ(1, prox.arg_size_size());
  const int n = prox.arg_size(0).dim(0)*prox.arg_size(0).dim(1);

  ScaledZoneParams params;
  if (prox.prox_function_type() == ProxFunction::NORM_1) {
    params.alpha = Eigen::VectorXd::Constant(n, 1);
    params.beta = Eigen::VectorXd::Constant(n, 1);
    params.C = 0;
    params.M = 0;
  } else if (prox.prox_function_type() == ProxFunction::SUM_DEADZONE) {
    params.alpha = Eigen::VectorXd::Constant(n, 1);
    params.beta = Eigen::VectorXd::Constant(n, 1);
    params.C = 0;
    params.M = prox.scaled_zone_params().m();
  } else if (prox.prox_function_type() == ProxFunction::SUM_HINGE) {
    params.alpha = Eigen::VectorXd::Constant(n, 1);
    params.beta = Eigen::VectorXd::Constant(n, 0);
    params.C = 0;
    params.M = 0;
  } else if (prox.prox_function_type() == ProxFunction::SUM_QUANTILE) {
    BlockVector tmp;
    affine::BuildAffineOperator(
        prox.scaled_zone_params().alpha_expr(), "alpha", nullptr, &tmp);
    affine::BuildAffineOperator(
        prox.scaled_zone_params().beta_expr(), "beta", nullptr, &tmp);
    params.alpha = Promote(tmp("alpha"), n);
    params.beta = Promote(tmp("beta"), n);
    params.C = 0;
    params.M = 0;
  } else {
    LOG(FATAL) << "Unknown prox type: " << prox.prox_function_type();
  }

  return params;
}

Eigen::VectorXd ApplyScaledZoneProx(
    const ScaledZoneParams& params,
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& v) {
  // Convenience/readability
  const int n = v.rows();
  const Eigen::VectorXd& alpha = params.alpha;
  const Eigen::VectorXd& beta = params.beta;
  const double M = params.M;
  const double C = params.C;

  Eigen::VectorXd x = (v.array()-C).matrix();
  for (int i=0; i < n; i++) {
    if (std::fabs(x(i)) <= M)
      x(i) = x(i);
    else if (x(i) > M + lambda(i) * alpha(i))
      x(i) = x(i) - lambda(i) * alpha(i);
    else if (x(i) < -M - lambda(i) * beta(i))
      x(i) = x(i) + lambda(i) * beta(i);
    else if (x(i) > 0)
      x(i) = M;
    else
      x(i) = -M;
  }

  return x;
}

void ScaledZoneProx::Init(const ProxOperatorArg& arg) {
  VectorProx::Init(arg);
  params_ = GetParams(arg.prox_function());
}

void ScaledZoneProx::ApplyVector(
    const VectorProxInput& input,
    VectorProxOutput* output) {
  output->set_value(
      0, ApplyScaledZoneProx(params_, input.lambda_vec(), input.value_vec(0)));
}

REGISTER_PROX_OPERATOR(NORM_1, ScaledZoneProx);
REGISTER_PROX_OPERATOR(SUM_DEADZONE, ScaledZoneProx);
REGISTER_PROX_OPERATOR(SUM_HINGE, ScaledZoneProx);
REGISTER_PROX_OPERATOR(SUM_QUANTILE, ScaledZoneProx);

class ScaledZoneEpigraph : public VectorProx {
 public:
  void Init(const ProxOperatorArg& arg) override {
    VectorProx::Init(arg);
    params_ = GetParams(arg.prox_function());
  }

 protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override;

 private:
  ScaledZoneParams params_;
};


bool abs_cmp_descending(const double &x, const double &y) {
    double ax = std::fabs(x), ay = std::fabs(y);
    return ax > ay;
}

double key(double M, double alpha, double beta, double x) {
  if (x > 0)
    return (x-M) / alpha;
  else
    return (x+M) / beta;
}

void ScaledZoneEpigraph::ApplyVector(
    const VectorProxInput& input,
    VectorProxOutput* output) {
  const Eigen::VectorXd& v = input.value_vec(0);
  const double s = input.value(1);

  // Convenience/readability
  const int n = v.rows();
  const Eigen::VectorXd& alpha = params_.alpha;
  const Eigen::VectorXd& beta = params_.beta;
  const double M = params_.M;
  const double C = params_.C;

  Eigen::VectorXd vec_y = v;
  double *y = vec_y.data();
  int ny = n;

  // Filter and eval function
  double fval = 0;
  for(int i=0; i<ny; ) {
    y[i] = y[i] - C;
    if(std::fabs(y[i]) <= M || (y[i]>0 && alpha(i)==0) || (y[i]<0 && beta(i)==0)) {
            assert(ny >= 1);
      std::swap(y[i], y[--ny]);
    } else {
      if(y[i] > M)
        fval += alpha(i) * (y[i]-M);
      else if(y[i] < -M)
        fval += beta(i) * (-y[i]-M);

      y[i] = key(M, alpha(i), beta(i), y[i]);
      i++;
    }
  }

  if (fval <= s){
    output->set_value(0, v);
    output->set_value(1, s);
    return;
  }

  double acc = -s, div = 0;
  int l = 0, r = ny-1;
  for(int kk=0; kk<n; kk++) {
    if(l > r)
      break;
    int m = random() % (r-l+1) + l;
    double lam = std::fabs(y[m]);
    std::swap(y[r], y[m]);
    VLOG(2) << "lam = " << lam;
    // 3 way partition (Segwick), invariant: [eq less ? more eq]
    // [p,i)==less, (j,q]==more
    int p = l-1, q = r;
    int i = p, j = q;
    while(true) {
      i++;
      while(std::fabs(y[i]) < lam)
        i++;
      j--;
      while(std::fabs(y[j]) > lam){
        if(j==l)
          break;
        j--;
      }
      if(i>=j)
        break;
      std::swap(y[i], y[j]);
      if(std::fabs(y[i]) == lam) {
        p++;
        std::swap(y[i], y[p]);
      }
      if(std::fabs(y[j]) == lam) {
        q--;
        std::swap(y[j], y[q]);
      }
    }

    std::swap(y[i], y[r]);
    j = i-1;
    i = i+1;
    for(int k=l; k<p; k++) {
      std::swap(y[k], y[j]);
      j--;
    }
    for(int k=q+1; k<r; k++) {
      std::swap(y[k], y[i]);
      i++;
    }

    double dacc = 0, ddiv = 0;
    double eqacc = 0, eqdiv = 0;
    for(int k=j+1; k<r+1; k++) {
      if(std::fabs(y[k]) == lam) {
        if(y[k] > 0) {
          eqacc += alpha(k) * alpha(k) * y[k];
          eqdiv += alpha(k) * alpha(k);
        }
        if(y[k] < 0) {
          eqacc += beta(k) * beta(k) * (-y[k]);
          eqdiv += beta(k) * beta(k);
        }
      } else if(y[k] > 0) {
        dacc +=  alpha(k) * alpha(k) * y[k];
        ddiv += alpha(k) * alpha(k);
      } else if(y[k] < 0) {
        dacc += beta(k) * beta(k) * (-y[k]);
        ddiv += beta(k) * beta(k);
      }
    }

    double g = (acc+dacc) - lam*(1+ div+ddiv);
    if(g < 0) {
      r = j;
      acc += dacc + eqacc;
      div += ddiv + eqdiv;
    } else if(g > 0) {
      l = i;
    } else {
      break;
    }
  }
        
  double lam = acc/(div+1);

  output->set_value(
      0, ApplyScaledZoneProx(params_, Eigen::VectorXd::Constant(n, lam), v));
  output->set_value(1, s+lam);
}

REGISTER_EPIGRAPH_OPERATOR(NORM_1, ScaledZoneEpigraph);
REGISTER_EPIGRAPH_OPERATOR(SUM_DEADZONE, ScaledZoneEpigraph);
REGISTER_EPIGRAPH_OPERATOR(SUM_HINGE, ScaledZoneEpigraph);
REGISTER_EPIGRAPH_OPERATOR(SUM_QUANTILE, ScaledZoneEpigraph);
