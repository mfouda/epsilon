#ifndef EPSILON_PROX_ELEMENTWISE_H
#define EPSILON_PROX_ELEMENTWISE_H

#include "epsilon/prox/prox.h"

class ElementwiseProxOperator : public ProxOperator {
 public:
  virtual void Init(const ProxOperatorArg& arg);
  virtual BlockVector Apply(const BlockVector& v);

  virtual void InitElementwise(const Eigen::VectorXd& lambda) {}
  virtual Eigen::VectorXd ApplyElementwise(const Eigen::VectorXd& v) = 0;

 private:
  void InitArgs(const AffineOperator& f);
  void InitConstraints(const AffineOperator& f);

  std::string key_;
  BlockMatrix AT_;
  Eigen::VectorXd lambda_, a_, b_;
};

#endif  // EPSILON_PROX_ELEMENTWISE_H
