#ifndef EPSILON_PROX_ELEMENTWISE_H
#define EPSILON_PROX_ELEMENTWISE_H

#include "epsilon/prox/vector.h"

class ElementwiseProx : public VectorProx {
 public:
  void Init(const ProxOperatorArg& arg) override;
  BlockVector Apply(const BlockVector& v) override;

 private:
  void InitArgs(const AffineOperator& f);
  void InitConstraints(const AffineOperator& f);

  std::string key_;
  BlockMatrix AT_;
  Eigen::VectorXd lambda_, a_, b_;
};

#endif  // EPSILON_PROX_ELEMENTWISE_H
