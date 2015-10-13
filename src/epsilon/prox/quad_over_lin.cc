#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/prox/newton.h"
#include "epsilon/prox/ortho_invariant.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

// \sum_i p_i^2 / x_i, x_i > 0
class QuadOverLinProx : public ProxOperator {
public:
  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& vq) override {
    int n = vq.rows()/2;
    Eigen::VectorXd v = vq.head(n);
    Eigen::VectorXd q = vq.tail(n);
    Eigen::VectorXd x(n);
    Eigen::VectorXd p(n);

    for(int i=0; i<n; i++) {
      // make xi large enough
      double xi = 100 + std::abs(v(i));
      int iter = 0, max_iter = 100;
      for(; iter<max_iter; iter++) {
        double f = xi*xi*xi + (4*lambda_-v(i))*xi*xi
          + lambda_*(4*lambda_-v(i))*xi
          - 4*lambda_*lambda_*v(i)-q(i)*q(i);
        double F = 3*xi*xi + 2*(4*lambda_-v(i))*xi
          + lambda_*(4*lambda_-v(i));
        xi = xi - f/F;
        if(std::abs(f) <= 1e-3)
          break;
      }
      if(iter == max_iter) {
        VLOG(2) << "Newton fail to find cubic\n";
      }else if(xi <= 0) {
        VLOG(2) << "x <= 0\n";
      }
      x(i) = xi;
      p(i) = q(i) / (1+2*lambda_/xi);
    }

    Eigen::VectorXd xp(2*n);
    xp.head(n) = x;
    xp.tail(n) = q;

    return xp;
  }

private:
  double lambda_;
};
REGISTER_PROX_OPERATOR(QuadOverLinProx);

class MatrixFracProx : public OrthoInvariantProx {
public:
  MatrixFracProx() : OrthoInvariantProx(std::make_unique<QuadOverLinProx>(), true) {}
};
REGISTER_PROX_OPERATOR(MatrixFracProx);
