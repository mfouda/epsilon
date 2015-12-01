#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/elementwise.h"
#include "epsilon/vector/vector_util.h"

class SumKLDivProx : public ElementwiseProx {
public:
  void InitElementwise(const Eigen::VectorXd& lambda) override {
    lambda_ = arg.lambda();
  }
  Eigen::VectorXd ApplyElementwise(const Eigen::VectorXd& uv) override {
    VLOG(2) << "lambda = " << VectorDebugString(lambda_) << "\n";
    double eps = 1e-12;
    int n = uv.rows()/2;
    Eigen::VectorXd u = uv.head(n);
    Eigen::VectorXd v = uv.tail(n);
    Eigen::VectorXd p(n);
    Eigen::VectorXd q(n);

    for(int i=0; i<n; i++) {
      int iter = 0, max_iter=1000;
      double res = 0;
      q(i) = 0.5;
      for(; iter < max_iter; iter++) {
        if(q(i)-v(i)+lambda_(i) < eps)
          q(i) = std::max(v(i)-lambda_(i) + eps, eps);
        double f = q(i)*(q(i)-v(i)+lambda_(i))/lambda_(i) - u(i)
          + lambda_(i)*( std::log(q(i)-v(i)+lambda_(i)) - std::log(lambda_(i)));
        double F = (2*q(i)-v(i)+lambda_(i))/lambda_(i)
          + lambda_(i)/(q(i)-v(i)+lambda_(i));
        res = f;
        if(std::abs(res) < eps and q(i) > 0)
          break;

        if(std::abs(F) <= eps)
          F = 1e-6 * (F>0?1:-1);
        q(i) = q(i) - f/F;
        if(q(i) < eps)
          q(i) = eps;
      }
      if(iter == max_iter)
        VLOG(2) << "Newton does not converge for kl_div prox\n";
      VLOG(2) << "newton_iter = " << iter << ", f = " << res << "\n";
      p(i) = q(i)*(q(i)-v(i))/lambda_(i) + q(i);
      if(p(i) < eps)
        p(i) = eps;
    }

    Eigen::VectorXd pq(2*n);
    pq.head(n) = p;
    pq.tail(n) = q;

    return pq;
  }
protected:
  Eigen::VectorXd lambda_;
};
REGISTER_PROX_OPERATOR(SUM_KL_DIV, SumKLDivProx);

// class KLDivEpigraph : public KLDivProx {
// public:
//   virtual Eigen::VectorXd Apply(const Eigen::VectorXd& suv) override {
//     double eps = 1e-10;
//     double s = suv(0);
//     int n = (suv.rows()-1)/2;
//     Eigen::VectorXd u = suv.segment(1, n);
//     Eigen::VectorXd v = suv.tail(n);
//     Eigen::VectorXd uv = suv.tail(2*n);

//     Eigen::VectorXd pq(2*n);
//     int iter = 0, max_iter = 100;
//     double res = 0;
//     lambda_ = 1;
//     for(; iter<max_iter; iter++) {
//       Eigen::VectorXd g(2);
//       Eigen::MatrixXd h(2,2);

//       pq = KLDivProx::Apply(uv);
//       Eigen::VectorXd p = pq.head(n);
//       Eigen::VectorXd q = pq.tail(n);
//       double glam = -s - lambda_;
//       double hlam = -1;
//       for(int i=0; i<n; i++) {
//         glam += p(i)*std::log(p(i)/q(i)) - p(i) + q(i);
//         g(0) = std::log(p(i)/q(i));
//         g(1) = -p(i)/q(i)+1;
//         h(0,0) = 1/p(i);
//         h(0,1) = -1/q(i);
//         h(1,0) = h(0,1);
//         h(1,1) = p(i)/(q(i)*q(i));
//         h = Eigen::MatrixXd::Identity(2, 2) + lambda_ * h;

//         hlam -= (g.transpose() * h.inverse() * g);
//       }
//       res = glam;
//       if(std::abs(res) <= eps)
//         break;
//       lambda_ = lambda_ - glam/hlam;
//       if(lambda_ < eps)
//         lambda_ = 1e-6;
//     }
//     if(iter == max_iter)
//       VLOG(2) << "Newton method won't converge for implicit kl_div\n";
//     VLOG(2) << "Use " << iter << " Newton Iters, res = " << res << "\n";

//     Eigen::VectorXd tpq(1+2*n);
//     tpq(0) = lambda_ + s;
//     tpq.tail(2*n) = KLDivProx::Apply(uv);

//     return tpq;
//   }
// };
// REGISTER_PROX_OPERATOR(KLDivEpigraph);
