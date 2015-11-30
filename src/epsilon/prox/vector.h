#ifndef EPSILON_PROX_VECTOR_H
#define EPSILON_PROX_VECTOR_H

#include "epsilon/prox/prox.h"

class VectorProx : public ProxOperator {
 public:
  void Init(const ProxOperatorArg& arg) override;
  BlockVector Apply(const BlockVector& v) override;

  virtual void InitVector(int n, double lambda) {}
  virtual Eigen::VectorXd ApplyVector(const Eigen::VectorXd& v) = 0;

 private:
  void InitArgs(const AffineOperator& f);
  void InitConstraints(const AffineOperator& f);

  std::string key_;
  int n_;
  BlockMatrix AT_;
  double alpha_, lambda_;
  Eigen::VectorXd b_;
};

#endif  // EPSILON_PROX_VECTOR_H
