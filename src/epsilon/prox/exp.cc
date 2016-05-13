
#include "epsilon/prox/vector_prox.h"

class ExpEpigraph : public VectorProx {
 public:
  virtual void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output);
};
REGISTER_EPIGRAPH_OPERATOR(EXP, ExpEpigraph);

void ExpEpigraph::ApplyVector(
    const VectorProxInput& input,
    VectorProxOutput* output) {
  const Eigen::VectorXd& v = input.value_vec(0);
  const Eigen::VectorXd& s = input.value_vec(1);

  const int n = v.rows();
  Eigen::VectorXd x = v;
  Eigen::VectorXd t = s;
  Eigen::VectorXd l = Eigen::VectorXd::Constant(n, 1);

  int iter = 0;
  const int max_iter = 100;
  const double eps = 1e-12;
  for (; iter < max_iter; iter++) {
    // TODO(mwytock): Parameterize this so we can generalize this to accept
    // other functions.
    Eigen::VectorXd exp_x = x.array().exp();
    const Eigen::VectorXd& fx = exp_x;
    const Eigen::VectorXd& dfx = exp_x;
    const Eigen::VectorXd& ddfx = exp_x;

    Eigen::VectorXd r0 = (x - v).array() + l.array()*dfx.array();
    Eigen::VectorXd r1 = t - s - l;
    Eigen::VectorXd r2 = fx - t;

    VLOG(2) << "iter " << iter << "\n"
            << "r0: " << VectorDebugString(r0) << "\n"
            << "r1: " << VectorDebugString(r1) << "\n"
            << "r2: " << VectorDebugString(r2);

    if (r0.lpNorm<Eigen::Infinity>() < eps &&
        r1.lpNorm<Eigen::Infinity>() < eps &&
        r2.lpNorm<Eigen::Infinity>() < eps) {
      break;
    }

    Eigen::VectorXd h = 1 + l.array()*ddfx.array();
    const Eigen::VectorXd& d = dfx;

    // Solve the system of equations
    // [h  0  d][dx] = - [r0]
    // [0  1 -1][dt]     [r1]
    // [d -1  0][dl]     [r2]
    Eigen::VectorXd dl =
        (-d.array()*r0.array() + h.array()*(r1 + r2).array()) /
        (d.array()*d.array() + h.array());
    Eigen::VectorXd dx = -(d.array()*dl.array() + r0.array()) / h.array();
    Eigen::VectorXd dt = dl - r1;

    x += dx;
    t += dt;
    l += dl;
  }

  // Correction for the easy cases, inputs that were already feasible
  Eigen::VectorXd f_v = v.array().exp();
  for (int i = 0; i < n; i++) {
    if (f_v(i) <= s(i)) {
      x(i) = v(i);
      t(i) = s(i);
    }
  }

  output->set_value(0, x);
  output->set_value(1, t);
}
