#ifndef EPSILON_PROX_ORTHO_INVARIANT_H
#define EPSILON_PROX_ORTHO_INVARIANT_H

#include "epsilon/prox/vector_prox.h"

class OrthoInvariantProx : public VectorProx {
 public:
  OrthoInvariantProx(
      ProxFunction::Type eigen_prox_type,
      bool symmetric_part = false,
      bool add_residual = false,
      bool epigraph = false)
      : eigen_prox_type_(eigen_prox_type),
        symmetric_part_(symmetric_part),
        add_residual_(add_residual),
        epigraph_(epigraph) {}

  void Init(const ProxOperatorArg& arg) override;

 protected:
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) override;

 private:
  void InitEigenProx();

  Eigen::VectorXd ApplyEigenProx(const Eigen::VectorXd& v);
  void ApplyEigenEpigraph(
      const Eigen::VectorXd& v, double s,
      Eigen::VectorXd* x, double* t);

  ProxFunction::Type eigen_prox_type_;
  bool symmetric_part_, add_residual_, epigraph_;

  int m_, n_;
  std::unique_ptr<ProxOperator> eigen_prox_;
};

#endif  // EPSILON_PROX_ORTHO_INVARIANT_H
