#include "epsilon/prox/prox.h"

class ScaledZoneProx : public ProxOperator {
public:
  ScaledZoneProx(double alpha, double beta, double C, double M) 
          : alpha_(alpha), beta_(beta), C_(C), M_(M) {};

  virtual void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
  }
  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override;

protected:
  double key(double x);

  double lambda_;
  double alpha_, beta_, C_, M_;
};

class ScaledZoneEpigraph : public ScaledZoneProx {
public:
  ScaledZoneEpigraph(double alpha, double beta, double C, double M)
          : ScaledZoneProx(alpha, beta, C, M) {};
  Eigen::VectorXd Apply(const Eigen::VectorXd& sv) override;
};
