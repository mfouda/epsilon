#include "epsilon/prox/prox.h"

class SmoothFunction {
 public:
  virtual double eval(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd gradf(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd hessf(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd proj_feasible(const Eigen::VectorXd& x) const {
    return x;
  }
};

class NewtonProx : public ProxOperator {
 public:
  NewtonProx(std::unique_ptr<SmoothFunction> f) : f_(std::move(f)) {}

  virtual void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
  }
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& v) override;

 protected:
  Eigen::VectorXd residual(
      const Eigen::VectorXd& x, const Eigen::VectorXd& v, double lam);

  std::unique_ptr<SmoothFunction> f_;
  double lambda_;
};

class NewtonEpigraph : public ProxOperator {
 public:
  NewtonEpigraph(std::unique_ptr<SmoothFunction> f) : f_(std::move(f)) {}

  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& v) override;

 private:
  Eigen::VectorXd residual(
      const Eigen::VectorXd& x, double t, double lam,
      const Eigen::VectorXd& v, double s);
  Eigen::VectorXd SolveArrowheadSystem(
      const Eigen::VectorXd& d, const Eigen::VectorXd& z, double alpha,
      const Eigen::VectorXd& b);

  std::unique_ptr<SmoothFunction> f_;
};

class ImplicitNewtonEpigraph : public NewtonProx {
 public:
  ImplicitNewtonEpigraph
    (std::unique_ptr<SmoothFunction> f) : NewtonProx(std::move(f)) {}

  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& v) override;
};
