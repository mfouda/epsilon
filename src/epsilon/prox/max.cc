
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/vector_prox.h"
#include "epsilon/vector/vector_util.h"

// max_i x_i
class MaxProx final : public VectorProx {
protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override {
    const Eigen::VectorXd& v = input.value_vec(0);
    const double lambda = input.lambda();

    int n = v.rows();
    Eigen::VectorXd y_vec = v;
    double *y = y_vec.data();
    sort(y, y+n, std::greater<double>());

    // lambda = \sum_i (v_i-t)_+ = \sum_{vi>=t} (v_i-t)
    //    x_i = min(t, v_i)

    double t = 0;
    double acc = -lambda, div = 0;
    for(int i=0; i<n; i++) {
      if(y[i]*div < acc)
        break;
      acc += y[i];
      div += 1;
      t = acc / div;
    }

    // VLOG(1) << "t = " << t << "\n";

    Eigen::VectorXd x(n);
    for(int i=0; i<n; i++)
      x(i) = std::min(v(i), t);

    output->set_value(0, x);
  }
};
REGISTER_PROX_OPERATOR(MAX, MaxProx);

// private:
//   double lambda;
// };
// REGISTER_PROX_OPERATOR(MaxEntriesProx);

// // [max_i x_i <= t]
// class MaxEntriesEpigraph final : public ProxOperator {
// public:
//   void Init(const ProxOperatorArg& arg) override {}
//   Eigen::VectorXd Apply(const Eigen::VectorXd& sv) override {
//     int n = sv.rows()-1;
//     double s = sv(0);
//     Eigen::VectorXd y_vec = sv.tail(n);

//     double *y = y_vec.data();
//     sort(y, y+n, std::greater<double>());

//     if(s >= y[0]){
//       return sv;
//     }
//     // delta = \sum_i (v_i-s-delta)_+
//     //    x_i = v-(v_i-s-delta)_+
//     //    t = s + delta
//     double delta = 0;
//     double acc = 0, div = 1;
//     for(int i=0; i<n; i++) {
//       if(div * (y[i]-s) < acc)
//         break;
//       acc += y[i]-s;
//       div += 1;
//       delta = acc/div;
//     }

//     double t = s + delta;
//     Eigen::VectorXd tx(n+1);
//     tx(0) = t;
//     for(int i=1; i<n+1; i++)
//       tx(i) = sv(i) - std::max(0., sv(i)-t);

//     return tx;
//   }
// };
// REGISTER_PROX_OPERATOR(MaxEntriesEpigraph);

// class LambdaMaxEpigraph final : public OrthoInvariantEpigraph {
// public:
//   LambdaMaxEpigraph() : OrthoInvariantEpigraph(std::make_unique<MaxEntriesEpigraph>(), true) {}
// };
// REGISTER_PROX_OPERATOR(LambdaMaxEpigraph);
