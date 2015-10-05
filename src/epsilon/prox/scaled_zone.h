#include "epsilon/prox/prox.h"

// Assumption: alpha,beta,M>=0, C in R
// f(x) = \sum_i \alpha \max(0, (x_i-C)-M) + \beta \max(0, -(x_i-C)-M)
class ScaledZoneProx : public ProxOperator {
public:
  ScaledZoneProx() : params_from_proto_(true) {}
  ScaledZoneProx(double alpha, double beta, double C, double M)
      : alpha_(alpha), beta_(beta), C_(C), M_(M),
        params_from_proto_(false) {}

  virtual void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    if (params_from_proto_) {
      alpha_ = arg.f_expr().proximal_operator().scaled_zone_params().alpha();
      beta_ = arg.f_expr().proximal_operator().scaled_zone_params().beta();
      C_ = arg.f_expr().proximal_operator().scaled_zone_params().c();
      M_ = arg.f_expr().proximal_operator().scaled_zone_params().m();
    }
  }
  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override;

protected:
  double key(double x);

  double lambda_;
  double alpha_, beta_, C_, M_;
  bool params_from_proto_;
};

class ScaledZoneEpigraph : public ScaledZoneProx {
public:
  ScaledZoneEpigraph() {}
  ScaledZoneEpigraph(double alpha, double beta, double C, double M)
          : ScaledZoneProx(alpha, beta, C, M) {}

  Eigen::VectorXd Apply(const Eigen::VectorXd& sv) override;
};
