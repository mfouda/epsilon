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

  // Find the largest cubic root of a depressed cubic
  // by Durand-Kerner method
  // x^3 + bx^2 + cx + d = 0
  double LargestRealCubicRoot(double b, double c, double d) {
    double eps = 1e-12;
    std::complex<double> p(0.4, 0.9), q=p*p, r=p*p*p;
    int iter = 0, max_iter = 100;
    for(; iter < max_iter; iter++) {
      std::complex<double> fp = p*p*p + b*p*p + c*p + d;
      std::complex<double> np = p - fp / ((p-q)*(p-r));
      std::complex<double> fq = q*q*q + b*q*q + c*q + d;
      std::complex<double> nq = q - fq / ((q-p)*(q-r));
      std::complex<double> fr = r*r*r + b*r*r + c*r + d;
      std::complex<double> nr = r - fr / ((r-p)*(r-q));
      if(std::abs(fp)<eps and std::abs(fq)<eps and std::abs(fr)<eps)
        break;
      //VLOG(2) << "fp = " << fp << ", fq = " << fq << ", fr = " << fr << "\n";
      p = np;
      q = nq;
      r = nr;
    }
    if(iter == max_iter)
      VLOG(2) << "Cubic failed\n";
    double m = -1e41;
    VLOG(2) << "p = " << p << ", q = " << q << ", r = " << r << "\n";
    if(std::abs(std::imag(p))<eps and std::real(p) > m)
      m = std::real(p);
    if(std::abs(std::imag(q))<eps and std::real(q) > m)
      m = std::real(q);
    if(std::abs(std::imag(r))<eps and std::real(r) > m)
      m = std::real(r);
    VLOG(2) << "Cubic iter = " << iter << ", f = " << m*m*m+b*m*m+c*m+d << "\n";
    return m;
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& qv) override {
    VLOG(2) << "lambda  = " << lambda_ << "\n";
    int n = qv.rows()/2;
    Eigen::VectorXd q = qv.head(n);
    Eigen::VectorXd v = qv.tail(n);
    Eigen::VectorXd p(n);
    Eigen::VectorXd x(n);

    for(int i=0; i<n; i++) {
      double xi = LargestRealCubicRoot(
          4*lambda_-v(i), 
          4*lambda_*(lambda_-v(i)), 
          -lambda_*(4*lambda_*v(i)+q(i)*q(i)));
      double pi =  q(i) / (1+2*lambda_/xi);

      double res1 = xi - v(i) - lambda_*pi*pi/(xi*xi);
      double res2 = pi - q(i) + 2*lambda_*pi/xi;
      VLOG(2) << "xi = " << xi << ", res1 = " << res1 << ", res2 = " << res2 << "\n";
      VLOG(2) << "lamba, v, q" << lambda_ << " " << v << " " << q << "\n";

      x(i) = xi;
      p(i) = pi;
    }

    Eigen::VectorXd px(2*n);
    px.head(n) = p;
    px.tail(n) = x;

    return px;
  }

private:
  double lambda_;
};
REGISTER_PROX_OPERATOR(QuadOverLinProx);

class MatrixFracProx : public QuadOverLinProx {
public:
  virtual void Init(const ProxOperatorArg& arg) override {
    QuadOverLinProx::Init(arg);
  }
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& qY) override { 
    int n = std::lround(-1+std::sqrt(1.+4*qY.rows()))/2; // n*(n+1) = #, n = (-1+(1+4*#)^0.5)/2

    Eigen::VectorXd q = qY.head(n);
    Eigen::VectorXd y = qY.tail(n*n);
    Eigen::MatrixXd Y = ToMatrix(y, n, n);
    Eigen::VectorXd d;
    Eigen::MatrixXd U, V;

    Y = (Y + Y.transpose()) / 2;

/*
    Eigen::JacobiSVD<Eigen::MatrixXd> solver(Y, Eigen::ComputeFullU | Eigen::ComputeFullV);
    d = solver.singularValues();
    U = solver.matrixU();
    V = solver.matrixV();
*/

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Y);
    CHECK_EQ(solver.info(), Eigen::Success);
    d = solver.eigenvalues();
    V = solver.eigenvectors();
    U = V;

    Eigen::VectorXd qbar = U.transpose() * q;

    Eigen::VectorXd qbard(2*n);
    qbard.head(n) = qbar;
    qbard.tail(n) = d;
    Eigen::VectorXd pbarx = QuadOverLinProx::Apply(qbard);
    Eigen::VectorXd pX(n+n*n);
    pX.head(n) = U * pbarx.head(n);
    Eigen::MatrixXd X = U * pbarx.tail(n).asDiagonal() * U.transpose();
    pX.tail(n*n) = ToVector(X);

/*
    VLOG(2) << "\npbar " << VectorDebugString(pbarx.head(n)) << "\nqbar " << VectorDebugString(qbar) \
      << "\nx " << pbarx.tail(n) << "\nd " << d << "\n";
*/

    return pX;
  }
};
REGISTER_PROX_OPERATOR(MatrixFracProx);
