#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

class ExpConeProx : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
  }
  Eigen::VectorXd Apply(const Eigen::VectorXd& uv) override {
    VLOG(2) << "lambda = " << lambda_ << "\n";
    double eps = 1e-13;
    int n = uv.rows()/2;
    Eigen::VectorXd pq(2*n);

    for(int i=0; i<n; i++) {
      int iter = 0, max_iter=1000;
      double res = 0;
      double lam = lambda_;
      // Let r = p/q;
      // Solve r(v+lam(r-1)e^r)-u+lam e^r = 0.
      // Then, p,q can be given by
      //   p-u+lam*e^r = 0,
      //   q-v+lam*e^r*(1-r) = 0.
      // find a feasible start such that q>0
      double r = 1;
      double ui = uv(i), vi = uv(n+i);
      while(vi+lam*std::exp(r)*(r-1) < 0)
        r *= 1.5;
      for(; iter < max_iter; iter++) {
        double er = std::exp(r);
        double f = r*(vi +lam*(r-1)*er) -ui +lam*er;
        double F = vi + lam*r*er*(1+r);
        res = f;
        if(std::abs(res) < eps)
          break;

        double beta = 0.001;
        double gamma = 0.5;
        double theta = 1;
        while(1) {
          double r_ = r - theta*f/F;
          double er_ = std::exp(r_);
          double res_ = r_*(vi +lam*(r_-1)*er_) -ui +lam*er_;
          double q = vi+lam*std::exp(r_)*(r_-1);
          if(q>0 and std::abs(res_) < (1-beta*theta)*res)
            break;
          theta *= gamma;
        }
      }
      if(iter == max_iter)
        VLOG(2) << "Newton does not converge for kl_div prox\n";
      VLOG(2) << "newton_iter = " << iter << ", f = " << res << "\n";
      pq(i) = ui-lam*std::exp(r);
      pq(n+i) = vi+lam*std::exp(r)*(r-1);
    }

    return pq;
  }
protected:
  double lambda_;
};

// I(q*exp(p/q) <= t), q>0
// using implicit Newton method
class ExpConeEpigraph final : public ExpConeProx {
  Eigen::VectorXd Apply(const Eigen::VectorXd& suv) override {
    return suv;
    double eps = 1e-10;
    int n = (suv.rows()-1)/2;
    double s = suv(0);
    Eigen::VectorXd u = suv.segment(1, n);
    Eigen::VectorXd v = suv.tail(n);

    int iter = 0, max_iter = 100;
    double res = 0;
    lambda_ = 1;

    for(; iter<max_iter; iter++) {
      Eigen::VectorXd pq = ExpConeProx::Apply(suv.segment(1,2*n+1));
      Eigen::VectorXd p = pq.head(n);
      Eigen::VectorXd q = pq.tail(n);
      double glam = -s - lambda_;
      double hlam = -1;
      for(int i=0; i<n; i++) {
        Eigen::Vector2d g;
        Eigen::Matrix2d h;

        double ri = p(i)/q(i);
        double epq = std::exp(ri);
        glam += q(i)*epq;
        g(0) = epq;
        g(1) = epq*(1-ri);
        double wi = lambda_/q(i)*epq;
        h(0,0) = 1+wi;
        h(0,1) = -wi*ri;
        h(1,0) = h(0,1);
        h(1,1) = 1+wi* ri*ri;
        h = Eigen::MatrixXd::Identity(2, 2) + lambda_ * h;

        hlam -= (g.transpose() * h.inverse() * g);
      }

      res = glam;
      if(std::abs(res) < eps or (lambda_<=eps*2 and glam/hlam > 0))
        break;
      lambda_ = lambda_ - glam/hlam;
      if(lambda_ < eps)
        lambda_ = eps;
    }
    Eigen::VectorXd tpq(2*n+1);
    tpq.tail(2*n) = ExpConeProx::Apply(suv.segment(1,2*n+1));
    tpq(1) = lambda_ + s;

    return tpq;
  }
};
REGISTER_PROX_OPERATOR(ExpConeEpigraph);
