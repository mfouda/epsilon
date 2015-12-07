#ifndef EPSILON_PROX_NEWTON_H
#define EPSILON_PROX_NEWTON_H

#include "epsilon/prox/vector_prox.h"

class SmoothFunction {
 public:
  virtual double eval(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd gradf(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd hessf(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd proj_feasible(const Eigen::VectorXd& x) const {
    return x;
  }
};

// TODO(mwytock): Mark these final and have the registration mechanism accept a
// lambda rather than a class.
class NewtonProx : public VectorProx {
public:
  NewtonProx(std::unique_ptr<SmoothFunction> f) : f_(std::move(f)) {}

protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override;

private:
  std::unique_ptr<SmoothFunction> f_;
};

class NewtonEpigraph : public VectorProx {
public:
  NewtonEpigraph(std::unique_ptr<SmoothFunction> f) : f_(std::move(f)) {}

protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override;

private:
  std::unique_ptr<SmoothFunction> f_;
};

class ImplicitNewtonEpigraph : public VectorProx {
 public:
  ImplicitNewtonEpigraph(std::unique_ptr<SmoothFunction> f) : f_(std::move(f)) {}

 protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override;

 private:
  std::unique_ptr<SmoothFunction> f_;
};

#endif  // EPSILON_PROX_NEWTON_H
