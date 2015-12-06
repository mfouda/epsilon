
#include <float.h>

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/vector_prox.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

// \sum_i log(xi)
class SumNegLogProx final : public VectorProx {
 protected:
  virtual void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override {
    const Eigen::VectorXd& v = input.value_vec(0);
    const Eigen::VectorXd& lambda = input.lambda_vec();
    const int n = v.rows();
    VectorXd x(n);

    for(int i=0; i<n; i++){
      if(v(i) >= 0)
        x(i) = (v(i) + std::sqrt(v(i)*v(i) + 4*lambda(i)))/2;
      else
        x(i) = 2*lambda(i) / (-v(i) + std::sqrt(v(i)*v(i) + 4*lambda(i)));
    }

    output->set_value(0, x);
  }
};
REGISTER_PROX_OPERATOR(SUM_NEG_LOG, SumNegLogProx);

// // I(-\sum_i log(xi) <= t)
// class NegativeLogEpigraph final : public NegativeLogProx {
// public:
//   void Init(const ProxOperatorArg& arg) override {};
//   Eigen::VectorXd Apply(const Eigen::VectorXd& sv) override {
//     int n = sv.rows()-1;
//     double s = sv(0);
//     Eigen::VectorXd v = sv.tail(n);

//     const double eps = 1e-10;
//     lambda_ = 1;

//     int iter = 0;
//     double g, h;
//     for(; iter<1000; iter++) {
//       g = -lambda_ -s;
//       h = -1;
//       for(int i=0; i<n; i++) {
//         // Avoiding catastrophic cancellation
//         double z = std::sqrt(v(i)*v(i) + 4*lambda_);
//         if(v(i) >= 0) { //
//           g -= std::log((v(i) + z)/2);
//           h -= 1. / (v(i)*(v(i)+z)/2 + 2*lambda_);
//         } else { // log x = -log 1/x
//           g += std::log((-v(i) + z)/(2*lambda_));
//           h -= 1. / (v(i)*2*lambda_/(-v(i)+z) + 2*lambda_);
//         }
//         // g -= std::log(v(i) + std::sqrt(v(i)*v(i)+4*lambda_));
//         // h -= 2./(v(i)*v(i) + 4*lambda_ + v(i)*std::sqrt(v(i)*v(i)+4*lambda_));
//       }

//       if(std::fabs(g) <= eps)
//         break;

//       if(h >= -1e-10)
//         h = -1e-10;

//       lambda_ -= g/h;
//       if(lambda_ <= 1e-10)
//         lambda_ = 1e-10;
//     }
//     VLOG(1) << iter+1 << " Newton iteration used on negative log epigraph." << "\n";
//     //VLOG(1) << "g = " << g << "\n";
//     //VLOG(1) << "lambda_ = " << lambda_ << "\n";

//     VectorXd tx(1+n);
//     double t = s + lambda_;
//     tx(0) = t;
//     tx.tail(n) = NegativeLogProx::Apply(sv.tail(n));

//     return tx;
//   }
// };
// REGISTER_PROX_OPERATOR(NegativeLogEpigraph);
