#include "epsilon/prox/ortho_invariant.h"

// Orthogonally Invariant Matrix Function satisfies
//      F(X) = F(UXV') = f(eigen_values(X))
Eigen::VectorXd OrthoInvariantProx::Apply(const Eigen::VectorXd& y) {
  int n = (int)std::sqrt(y.rows());

  Eigen::MatrixXd Y = ToMatrix(y, n, n);
  Eigen::VectorXd d;
  Eigen::MatrixXd U, V;
  if(to_symm_) {
    Y = (Y + Y.transpose()) / 2;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Y);
    CHECK_EQ(solver.info(), Eigen::Success);
    d = solver.eigenvalues();
    V = solver.eigenvectors();
    U = V;
  } else {
    Eigen::JacobiSVD<Eigen::MatrixXd> solver(Y, Eigen::ComputeThinU | Eigen::ComputeThinV);
    d = solver.singularValues();
    U = solver.matrixU();
    V = solver.matrixV();
  }

  VLOG(1) << "\nD = " << VectorDebugString(d) << "\n";

  Eigen::VectorXd x_tilde = f_->Apply(d);

  Eigen::MatrixXd X = U*x_tilde.asDiagonal()*V.transpose();

  return ToVector(X);
}


Eigen::VectorXd OrthoInvariantEpigraph::Apply(const Eigen::VectorXd& sy) {
    int n = (int)std::sqrt(sy.rows()-1);

    double s = sy(0);
    Eigen::MatrixXd Y = ToMatrix(sy.tail(n*n), n, n);
    Eigen::VectorXd d;
    Eigen::MatrixXd U, V;
    if(to_symm_) {
      Y = (Y + Y.transpose()) / 2;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Y);
      CHECK_EQ(solver.info(), Eigen::Success);
      d = solver.eigenvalues();
      U = solver.eigenvectors();
      V = U;
    } else {
      Eigen::JacobiSVD<Eigen::MatrixXd> solver(Y, Eigen::ComputeThinU | Eigen::ComputeThinV);
      d = solver.singularValues();
      U = solver.matrixU();
      V = solver.matrixV();
    }

    VLOG(1) << "\nD = " << VectorDebugString(d) << "\n";

    Eigen::VectorXd f_sv(1+n);
    f_sv(0) = s;
    f_sv.tail(n) = d;
    Eigen::VectorXd f_tx = f_->Apply(f_sv);

    double t = f_tx(0);
    Eigen::VectorXd x_tilde = f_tx.tail(n);
    Eigen::MatrixXd X = U*x_tilde.asDiagonal()*V.transpose();
    VectorXd tx(1+n*n);
    tx(0) = t;
    tx.tail(n*n) = ToVector(X);

    return tx;
}
