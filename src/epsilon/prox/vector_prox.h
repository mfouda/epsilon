#ifndef EPSILON_PROX_VECTOR_H
#define EPSILON_PROX_VECTOR_H

#include "epsilon/prox/prox.h"

class VectorProxInput {
 public:
  double lambda() const;
  const Eigen::VectorXd& lambda_vec() const;
  const Eigen::MatrixXd& lambda_mat() const;

  double value(int i) const;
  const Eigen::VectorXd& value_vec(int i) const;
  const Eigen::MatrixXd& value_mat(int i) const;
};

class VectorProxOutput {
 public:
  void set_value(int i, double x);
  void set_value(int i, const Eigen::VectorXd& x);
  void set_value(int i, const Eigen::MatrixXd& X);
};

class VectorProx : public ProxOperator {
 public:
  void Init(const ProxOperatorArg& arg) override;
  BlockVector Apply(const BlockVector& v) override;

  // To be overriden by subclasses
  virtual void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) = 0;

 private:
  void InitArgs(const AffineOperator& f);
  void InitConstraints(const AffineOperator& f);

  void ProcessInput(const BlockVector& v);
  BlockVector ProcessOutput();

  std::string key_;
  int n_;
  BlockMatrix AT_;
  double alpha_, lambda_;
  Eigen::VectorXd b_;

  // Used in ApplyVector()
  // NOTE(mwytock): Storing per-Apply() state like this makes VectorProx not
  // threadsafe which may be a consideration in future
  VectorProxInput input_;
  VectorProxOutput output_;
};

#endif  // EPSILON_PROX_VECTOR_H
