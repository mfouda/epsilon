#ifndef EPSILON_PROX_MAX_H
#define EPSILON_PROX_MAX_H

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/prox/elementwise.h"
#include "epsilon/vector/vector_util.h"

// max_i x_i
class MaxProx final : public VectorProx {
public:
  void InitVector(int n, double lambda) override {
    lambda_ = lambda;
  }

  Eigen::VectorXd ApplyVector(const Eigen::VectorXd& v) override;
private:
  double lambda_;
};

#endif // EPSILON_PROX_MAX_H
