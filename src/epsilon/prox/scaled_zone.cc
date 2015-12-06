#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/vector_prox.h"
#include "epsilon/vector/vector_util.h"

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
  ProxFunction::ScaledZoneParams params_;
};

ProxFunction::ScaledZoneParams GetParams(const ProxFunction& prox) {
  ProxFunction::ScaledZoneParams params = prox.scaled_zone_params();
  if (prox.prox_function_type() == ProxFunction::NORM_1) {
    params.set_alpha(1);
    params.set_beta(1);
    params.set_c(0);
    params.set_m(0);
  } else if (prox.prox_function_type() == ProxFunction::SUM_DEADZONE) {
    // Use existing m
    params.set_alpha(1);
    params.set_beta(1);
    params.set_c(0);
  } else if (prox.prox_function_type() == ProxFunction::SUM_HINGE) {
    params.set_alpha(1);
    params.set_beta(0);
    params.set_c(0);
    params.set_m(0);
  } else if (prox.prox_function_type() == ProxFunction::SUM_QUANTILE) {
    // Use existing alpha/beta
    params.set_m(0);
    params.set_c(0);
  } else {
    LOG(FATAL) << "Unknown prox type: " << prox.prox_function_type();
  }

  return params;
}

Eigen::VectorXd ApplyScaledZoneProx(
    const ProxFunction::ScaledZoneParams& params,
    const Eigen::VectorXd& lambda,
    const Eigen::VectorXd& v) {
  // Convenience/readability
  const int n = v.rows();
  const double& alpha = params.alpha();
  const double& beta = params.beta();
  const double& M = params.m();
  const double& C = params.c();

  Eigen::VectorXd x = (v.array()-C).matrix();
  for (int i=0; i < n; i++) {
    if (std::fabs(x(i)) <= M)
      x(i) = x(i);
    else if (x(i) > M + lambda(i) * alpha)
      x(i) = x(i) - lambda(i) * alpha;
    else if (x(i) < -M - lambda(i) * beta)
      x(i) = x(i) + lambda(i) * beta;
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
  ProxFunction::ScaledZoneParams params_;
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
  const double& alpha = params_.alpha();
  const double& beta = params_.beta();
  const double& M = params_.m();
  const double& C = params_.c();

  Eigen::VectorXd vec_y = v;
  double *y = vec_y.data();
  int ny = n;

  // Filter and eval function
  double fval = 0;
  for(int i=0; i<ny; ) {
    y[i] = y[i] - C;
    if(std::fabs(y[i]) <= M || (y[i]>0 && alpha==0) || (y[i]<0 && beta==0)) {
            assert(ny >= 1);
      std::swap(y[i], y[--ny]);
    } else {
      if(y[i] > M)
        fval += alpha * (y[i]-M);
      else if(y[i] < -M)
        fval += beta * (-y[i]-M);

      y[i] = key(M, alpha, beta, y[i]);
      i++;
    }
  }

  if (fval <= s){
    output->set_value(0, v);
    output->set_value(1, s);
    return;
  }

  std::sort(y, y+ny, abs_cmp_descending);

  double div = 0;
  double acc = -s;

  for(int i=0; i<ny; i++) {
    double lam = acc/(div+1);
    if(std::fabs(y[i]) <= lam)
      break;

    if(y[i] > 0) {
      acc = acc + alpha*alpha * y[i];
      div = div + alpha*alpha;
    } else if(y[i] < 0) {
      acc = acc + beta*beta * (-y[i]);
      div = div + beta*beta;
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
