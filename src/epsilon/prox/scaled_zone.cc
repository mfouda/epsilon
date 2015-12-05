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

void ScaledZoneProx::Init(const ProxOperatorArg& arg) {
  VectorProx::Init(arg);
  params_ = GetParams(arg.prox_function());
}

void ScaledZoneProx::ApplyVector(
    const VectorProxInput& input,
    VectorProxOutput* output) {
  const Eigen::VectorXd& lambda = input.lambda_vec();
  const Eigen::VectorXd& v = input.value_vec(0);

  // Convenience/readability
  const int n = v.rows();
  const double& alpha = params_.alpha();
  const double& beta = params_.beta();
  const double& M = params_.m();
  const double& C = params_.c();

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

  output->set_value(0, (x.array()+C).matrix());
}

REGISTER_PROX_OPERATOR(NORM_1, ScaledZoneProx);
REGISTER_PROX_OPERATOR(SUM_DEADZONE, ScaledZoneProx);
REGISTER_PROX_OPERATOR(SUM_HINGE, ScaledZoneProx);
REGISTER_PROX_OPERATOR(SUM_QUANTILE, ScaledZoneProx);

// bool abs_cmp_descending(const double &x, const double &y)
// {
//     double ax = std::fabs(x), ay = std::fabs(y);
//     return ax > ay;
// }

// double ScaledZoneProx::key(double x) {
//   if(x>0)
//     return (x-M_) / alpha_;
//   else
//     return (x+M_) / beta_;
// }

// Eigen::VectorXd ScaledZoneEpigraph::Apply(const Eigen::VectorXd& sv) {
//   const int n = sv.rows() - 1;
//   const double s = sv(0);
//   Eigen::VectorXd vec_y = sv.tail(n);
//   double *y = vec_y.data();
//   int ny = n;

//   // Filter and eval function
//   double fval = 0;
//   for(int i=0; i<ny; ) {
//     y[i] = y[i] - C_;
//     if(std::fabs(y[i]) <= M_ || (y[i]>0 && alpha_==0) || (y[i]<0 && beta_==0)) {
//             assert(ny >= 1);
//       std::swap(y[i], y[--ny]);
//     } else {
//       if(y[i] > M_)
//         fval += alpha_ * (y[i]-M_);
//       else if(y[i] < -M_)
//         fval += beta_ * (-y[i]-M_);

//       y[i] = key(y[i]);
//       i++;
//     }
//   }

//   if (fval <= s){
//     return sv;
//   }

//   std::sort(y, y+ny, abs_cmp_descending);

//   double div = 0;
//   double acc = -s;

//   for(int i=0; i<ny; i++) {
//     double lam = acc/(div+1);
//     if(std::fabs(y[i]) <= lam)
//       break;

//     if(y[i] > 0) {
//       acc = acc + alpha_*alpha_ * y[i];
//       div = div + alpha_*alpha_;
//     } else if(y[i] < 0) {
//       acc = acc + beta_*beta_ * (-y[i]);
//       div = div + beta_*beta_;
//     }
//   }
//   double lam = acc/(div+1);

//   Eigen::VectorXd tx(n+1);
//   this->lambda_ = lam;
//   tx(0) = s+lam;
//   tx.tail(n) = ScaledZoneProx::Apply(sv.tail(n));
//   return tx;
// }
// REGISTER_PROX_OPERATOR(ScaledZoneEpigraph);
