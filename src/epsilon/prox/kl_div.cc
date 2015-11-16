#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

class KLDivProx : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
  }
  Eigen::VectorXd Apply(const Eigen::VectorXd& uv) override {
    VLOG(2) << "lambda = " << lambda_ << "\n";
    double eps = 1e-13;
    int n = uv.rows()/2;
    Eigen::VectorXd u = uv.head(n);
    Eigen::VectorXd v = uv.tail(n);
    Eigen::VectorXd p(n);
    Eigen::VectorXd q(n);

    for(int i=0; i<n; i++) {
      int iter = 0, max_iter=1000;
      double res = 0;
      double qhat = std::max((0.5+lambda_-v(i))/lambda_, eps);
      if(std::abs(u(i)) < eps*eps and std::abs(v(i)) < eps*eps){
        p(i) = u(i);
        q(i) = v(i);
        continue;
      }
      // Solve: 
      //   q(lam-v+q)/lam-u+lam*log(lam-v+q)/lam = 0
      // Let qhat = (lam-v+q)/lam, then
      //   lam*qhat*qhat-(1-v/lam)qhat-u+lam*log(qhat)=0
      for(; iter < max_iter; iter++) {
        double f = lambda_*qhat*qhat + (v(i)-lambda_)*qhat - u(i)
            + lambda_*std::log(qhat);
        double F = 2*lambda_*qhat + (v(i)-lambda_) + lambda_/qhat;
        res = f/F;
        if(std::abs(res) < eps or (qhat <= eps*2 and f/F > 0) or (lambda_*qhat+v(i)-lambda_ <= eps*2 and f/F > 0))
          break;

        /*
        if(std::abs(F) < eps)
          F = eps * (F>0?1:-1);
        */
        qhat = qhat - f/F;
        if(qhat < eps)
          qhat = eps;
        if(lambda_*qhat+v(i)-lambda_ < eps)
          qhat = (eps+lambda_-v(i))/lambda_;
      }
      if(iter == max_iter)
        VLOG(2) << "Newton does not converge for kl_div prox\n";
      VLOG(2) << "newton_iter = " << iter << ", f = " << res << "\n";
      q(i) = lambda_*qhat + v(i)-lambda_;
      p(i) = q(i)*qhat;
      /*
      if(p(i) < eps)
        p(i) = eps;
      if(q(i) < eps)
        q(i) = eps;
      */
    }

    Eigen::VectorXd pq(2*n);
    pq.head(n) = p;
    pq.tail(n) = q;

    return pq;
  }
protected:
  double lambda_;
};
REGISTER_PROX_OPERATOR(KLDivProx);

class KLDivEpigraph : public KLDivProx {
public:
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& suv) override {
    double eps = 1e-10;
    double s = suv(0);
    int n = (suv.rows()-1)/2;
    Eigen::VectorXd u = suv.segment(1, n);
    Eigen::VectorXd v = suv.tail(n);
    Eigen::VectorXd uv = suv.tail(2*n);

    Eigen::VectorXd pq(2*n);
    int iter = 0, max_iter = 100;
    double res = 0;
    lambda_ = 1;
    for(; iter<max_iter; iter++) {
      Eigen::VectorXd g(2);
      Eigen::MatrixXd h(2,2);

      pq = KLDivProx::Apply(uv);
      Eigen::VectorXd p = pq.head(n);
      Eigen::VectorXd q = pq.tail(n);
      double glam = -s - lambda_;
      double hlam = -1;
      for(int i=0; i<n; i++) {
        glam += p(i)*std::log(p(i)/q(i)) - p(i) + q(i);
        g(0) = std::log(p(i)/q(i));
        g(1) = -p(i)/q(i)+1;
        h(0,0) = 1/p(i);
        h(0,1) = -1/q(i);
        h(1,0) = h(0,1);
        h(1,1) = p(i)/(q(i)*q(i));
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
    if(iter == max_iter)
      VLOG(2) << "Newton method won't converge for implicit kl_div\n";
    VLOG(2) << "Use " << iter << " Newton Iters, res = " << res << "\n";

    Eigen::VectorXd tpq(1+2*n);
    tpq(0) = lambda_ + s;
    tpq.tail(2*n) = KLDivProx::Apply(uv);

    return tpq;
  }
};
REGISTER_PROX_OPERATOR(KLDivEpigraph);
