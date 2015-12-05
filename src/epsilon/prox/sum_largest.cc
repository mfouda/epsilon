#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/vector_prox.h"
#include "epsilon/vector/vector_util.h"

class SumLargestProx : public VectorProx {
 public:
  void Init(const ProxOperatorArg& arg) override {
    VectorProx::Init(arg);
    k_ = arg.prox_function().sum_largest_params().k();
  }

 protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override {
    const double lambda = input.lambda();
    const Eigen::VectorXd& v = input.value_vec(0);
    const int n = v.rows();
    Eigen::VectorXd y_vec = v;
    double *y = y_vec.data();
    sort(y, y+n, std::greater<double>());

    // k *lam = \sum_i (y_i - q)_{0, lam}
    // q = (#{y_i >= q+lam} lam +\sum_{q <= y_i < q+lam} y_i - k * lam)
    //     / #{q <= y_i < q+lam}
    // inside: the window of [q, q+lam)
    // initialize with nothing inside
    double q = 0;
    double acc = - k_*lambda;
    int inside = 0;
    int i=0, j=0;
    for(; i<n and j<n; ) {
      //VLOG(2) << "acc=" <<  acc << ", yi = " << y[i] << ", yj = " << y[j] << "\n";
      // yi < q and yj < q+lambda
      if(y[i]*inside <= acc and (y[j]-lambda)*inside <= acc)
        break;
      // front-yi <= rear - yj, where rear-front=lam
      if(y[i] >= y[j]-lambda) {
        acc += y[i];
        inside += 1;
        i++;
      } else {
        acc += -y[j]+lambda;
        inside -= 1;
        j++;
      }
      q = acc / inside;
    }
    Eigen::VectorXd x(n);
    for(int i=0; i<n; i++) {
      x(i) = v(i) - std::max(0., std::min(lambda, v(i)-q));
    }

    output->set_value(0, x);
  }

 private:
  int k_;
};
REGISTER_PROX_OPERATOR(SUM_LARGEST, SumLargestProx);
