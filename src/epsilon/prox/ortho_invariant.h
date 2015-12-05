#ifndef EPSILON_PROX_ORTHO_INVARIANT_H
#define EPSILON_PROX_ORTHO_INVARIANT_H

#include "epsilon/prox/prox.h"

class OrthoInvariantProx : public ProxOperator {
 public:
  OrthoInvariantProx(
      ProxFunction::Type eigen_prox_type,
      bool symmetric = false,
      bool add_non_symmetric = false)
      : eigen_prox_type_(eigen_prox_type),
        symmetric_(symmetric),
        add_non_symmetric_(add_non_symmetric) {}

  void Init(const ProxOperatorArg& arg) override;
  BlockVector Apply(const BlockVector& v) override;

 private:
  void InitArgs(const AffineOperator& f);
  void InitConstraints(const AffineOperator& f);
  void InitEigenProx();
  Eigen::MatrixXd ApplyOrthoInvariant(const Eigen::MatrixXd& Y);
  Eigen::VectorXd ApplyEigenProx(const Eigen::VectorXd& v);

  ProxFunction::Type eigen_prox_type_;
  bool symmetric_, add_non_symmetric_;

  int m_, n_;
  std::string key_;
  BlockMatrix AT_;
  double lambda_;
  Eigen::MatrixXd B_;
  std::unique_ptr<ProxOperator> eigen_prox_;
};

// class OrthoInvariantEpigraph: public ProxOperator {
// public:
//   virtual void Init(const ProxOperatorArg& arg) override {
//     lambda_ = arg.lambda();
//     ProxOperatorArg prox_arg(lambda_, NULL, NULL, NULL);
//     f_->Init(prox_arg);
//   }
//   OrthoInvariantEpigraph(std::unique_ptr<ProxOperator> f, bool symm_part_=false)
//     : f_(std::move(f)), symm_part_(symm_part_) {}
//   virtual Eigen::VectorXd Apply(const Eigen::VectorXd& sy) override;

// protected:
//   std::unique_ptr<ProxOperator> f_;
//   double lambda_;
//   bool symm_part_;
// };

#endif  // EPSILON_PROX_ORTHO_INVARIANT_H
