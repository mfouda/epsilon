#include "epsilon/prox/prox.h"

// Orthogonally Invariant Matrix Function satisfies
//      F(X) = F(UXV') = f(eigen_values(X))
class OrthoInvariantProx: public ProxOperator {
public:
  virtual void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    ProxOperatorArg prox_arg(lambda_, NULL, NULL);
    f_->Init(prox_arg);
  }
  OrthoInvariantProx(std::unique_ptr<ProxOperator> f) : f_(std::move(f)) {}
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    int n = (int)std::sqrt(v.rows());

    Eigen::MatrixXd V = ToMatrix(v, n, n);
    V = (V + V.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(V);
    CHECK_EQ(solver.info(), Eigen::Success);
    const Eigen::VectorXd& d = solver.eigenvalues();
    const Eigen::MatrixXd& U = solver.eigenvectors();

    Eigen::VectorXd x_tilde = f_->Apply(d);

    Eigen::MatrixXd X = U*x_tilde.asDiagonal()*U.transpose();

    return ToVector(X);
  }

protected:
  std::unique_ptr<ProxOperator> f_;
  double lambda_;
};

class OrthoInvariantEpigraph: public ProxOperator {
public:
  virtual void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    ProxOperatorArg prox_arg(lambda_, NULL, NULL);
    f_->Init(prox_arg);
  }
  OrthoInvariantEpigraph(std::unique_ptr<ProxOperator> f) : f_(std::move(f)) {}
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& sv) override {
    int n = (int)std::sqrt(sv.rows()-1);

    double s = sv(0);
    Eigen::MatrixXd V = ToMatrix(sv.tail(n*n), n, n);
    V = (V + V.transpose())/2;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(V);
    CHECK_EQ(solver.info(), Eigen::Success);
    const Eigen::VectorXd& d = solver.eigenvalues();
    const Eigen::MatrixXd& U = solver.eigenvectors();

    Eigen::VectorXd f_sv(1+n);
    f_sv(0) = s;
    f_sv.tail(n) = d;
    Eigen::VectorXd f_tx = f_->Apply(f_sv);

    double t = f_tx(0);
    Eigen::VectorXd x_tilde = f_tx.tail(n);
    Eigen::MatrixXd X = U*x_tilde.asDiagonal()*U.transpose();
    VectorXd tx(1+n*n);
    tx(0) = t;
    tx.tail(n*n) = ToVector(X);

    return tx;
  }

protected:
  std::unique_ptr<ProxOperator> f_;
  double lambda_;
};
