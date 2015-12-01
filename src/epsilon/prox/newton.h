#ifndef EPSILON_PROX_NEWTON_H
#define EPSILON_PROX_NEWTON_H

#include "epsilon/prox/elementwise.h"

class SmoothFunction {
 public:
  virtual double eval(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd gradf(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd hessf(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd proj_feasible(const Eigen::VectorXd& x) const {
    return x;
  }
};

class NewtonProx : public ElementwiseProx {
 public:
  NewtonProx(std::unique_ptr<SmoothFunction> f) : f_(std::move(f)) {}

 protected:
  Eigen::VectorXd ApplyElementwise(
      const Eigen::VectorXd& lambda,
      const Eigen::VectorXd& v) override;

  Eigen::VectorXd residual(
      const Eigen::VectorXd& x,
      const Eigen::VectorXd& v,
      const Eigen::VectorXd& lambda);

  std::unique_ptr<SmoothFunction> f_;
  Eigen::VectorXd lambda_;
};

// class NewtonEpigraph : public ProxOperator {
//  public:
//   NewtonEpigraph(std::unique_ptr<SmoothFunction> f) : f_(std::move(f)) {}

//   virtual Eigen::VectorXd Apply(const Eigen::VectorXd& v) override;

//  private:
//   Eigen::VectorXd residual(
//       const Eigen::VectorXd& x, double t, double lam,
//       const Eigen::VectorXd& v, double s);
//   Eigen::VectorXd SolveArrowheadSystem(
//       const Eigen::VectorXd& d, const Eigen::VectorXd& z, double alpha,
//       const Eigen::VectorXd& b);

//   std::unique_ptr<SmoothFunction> f_;
// };

// class ImplicitNewtonEpigraph : public NewtonProx {
//  public:
//   ImplicitNewtonEpigraph
//     (std::unique_ptr<SmoothFunction> f) : NewtonProx(std::move(f)) {}

//   virtual Eigen::VectorXd Apply(const Eigen::VectorXd& v) override;
// };

#endif  // EPSILON_PROX_NEWTON_H
