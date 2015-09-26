#include "epsilon/prox/prox.h"

class NewtonProx : public ProxOperator {
public:
  virtual double f(const Eigen::VectorXd &x) = 0;
  virtual Eigen::VectorXd gradf(const Eigen::VectorXd &x) = 0;
  virtual Eigen::VectorXd hessf(const Eigen::VectorXd &x) = 0;

  Eigen::VectorXd residual(const Eigen::VectorXd &x, const Eigen::VectorXd &v, double lam);
  Eigen::VectorXd ProxByNewton(const Eigen::VectorXd &v, double lam);
};

class NewtonEpigraph : public ProxOperator {
public:
  virtual double f(const Eigen::VectorXd &x) = 0;
  virtual Eigen::VectorXd gradf(const Eigen::VectorXd &x) = 0;
  virtual Eigen::VectorXd hessf(const Eigen::VectorXd &x) = 0;

  Eigen::VectorXd residual
    (const Eigen::VectorXd &x, double t, double lam, const Eigen::VectorXd &v, double s);
  Eigen::VectorXd EpiByNewton(const Eigen::VectorXd &v, double s);
  Eigen::VectorXd solve_arrowhead_system
    (const Eigen::VectorXd &d, const Eigen::VectorXd &z, double alpha,
     const Eigen::VectorXd &b);
};
