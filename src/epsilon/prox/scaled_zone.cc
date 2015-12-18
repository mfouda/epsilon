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

  std::sort(y, y+ny, abs_cmp_descending);

  double div = 0;
  double acc = -s;

  for(int i=0; i<ny; i++) {
    double lam = acc/(div+1);
    if(std::fabs(y[i]) <= lam)
      break;

    if(y[i] > 0) {
      acc = acc + alpha(i)*alpha(i) * y[i];
      div = div + alpha(i)*alpha(i);
    } else if(y[i] < 0) {
      acc = acc + beta(i)*beta(i) * (-y[i]);
      div = div + beta(i)*beta(i);
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
