#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

class SumLargestProx : public ProxOperator {
  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    k_ = arg.f_expr().k();
    //VLOG(2) << "k = " << k_ << "\n";
  }
  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    int n = v.rows();
    Eigen::VectorXd y_vec = v;
    double *y = y_vec.data();
    sort(y, y+n, std::greater<double>());

    // k *lam = \sum_i (y_i - q)_{0, lam}
    // q = (#{y_i >= q+lam} lam +\sum_{q <= y_i < q+lam} y_i - k * lam)
    //     / #{q <= y_i < q+lam}
    // inside: the window of [q, q+lam)
    // initialize with nothing inside
    double q = 0;
    double acc = - k_*lambda_;
    int inside = 0;
    int i=0, j=0;
    for(; i<n and j<n; ) {
      //VLOG(2) << "acc=" <<  acc << ", yi = " << y[i] << ", yj = " << y[j] << "\n";
      // yi < q and yj < q+lambda_
      if(y[i]*inside <= acc and (y[j]-lambda_)*inside <= acc)
        break;
      // front-yi <= rear - yj, where rear-front=lam
      if(y[i] >= y[j]-lambda_) {
        acc += y[i];
        inside += 1;
        i++;
      } else {
        acc += -y[j]+lambda_;
        inside -= 1;
        j++;
      }
      q = acc / inside;
    }
    Eigen::VectorXd x(n);
    for(int i=0; i<n; i++) {
      x(i) = v(i) - std::max(0., std::min(lambda_, v(i)-q));
    }

    return x;
  }
private:
  double lambda_;
  int k_;
};
REGISTER_PROX_OPERATOR(SumLargestProx);
