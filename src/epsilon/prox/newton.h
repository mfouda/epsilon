#ifndef EPSILON_PROX_NEWTON_H
#define EPSILON_PROX_NEWTON_H

#include "epsilon/prox/vector_prox.h"

class SmoothFunction {
 public:
  virtual double eval(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd gradf(const Eigen::VectorXd& x) const = 0;
  // return (I+lambda*H(x))^{-1}v
  virtual Eigen::VectorXd hess_inv(const Eigen::VectorXd& lambda,
      const Eigen::VectorXd& x, const Eigen::VectorXd& v) const = 0;
  virtual Eigen::VectorXd hess_inv(double lambda,
      const Eigen::VectorXd& x, const Eigen::VectorXd& v) const {
    int n = x.rows();
    return hess_inv(Eigen::VectorXd::Constant(n, lambda), x, v);
  }
  virtual Eigen::VectorXd proj_feasible(const Eigen::VectorXd& x) const {
    return x;
  }
};

class ElemwiseSmoothFunction : public SmoothFunction {
 public:
  virtual Eigen::VectorXd hessf(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd hess_inv(const Eigen::VectorXd& lambda,
      const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override {
    int n = x.rows();
    Eigen::VectorXd hx = Eigen::VectorXd::Constant(n, 1.) + lambda.asDiagonal()*hessf(x);
    return (v.array() / hx.array()).matrix();
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

class ImplicitGradientEpigraph : public VectorProx {
 public:
  ImplicitGradientEpigraph(std::unique_ptr<VectorProx> f) : f_(std::move(f)) {}

 protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override;

 private:
  std::unique_ptr<VectorProx> f_;
};

#endif  // EPSILON_PROX_NEWTON_H
