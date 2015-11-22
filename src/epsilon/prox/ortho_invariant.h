
#include "epsilon/prox/elementwise.h"
#include "epsilon/prox/prox.h"

class OrthoInvariantProx : public ProxOperator {
public:
  OrthoInvariantProx(
      std::unique_ptr<ElementwiseProx> eigen_prox,
      bool symmetric = false,
      bool add_non_symmetric = false)
    : eigen_prox_(std::move(eigen_prox)),
      symmetric_(symmetric),
      add_non_symmetric_(add_non_symmetric) {}

  void Init(const ProxOperatorArg& arg) override;
  BlockVector Apply(const BlockVector& v) override;

private:
  void InitArgs(const AffineOperator& f);
  void InitConstraints(const AffineOperator& f);
  Eigen::MatrixXd ApplyOrthoInvariant(const Eigen::MatrixXd& Y);

  std::unique_ptr<ElementwiseProx> eigen_prox_;
  bool symmetric_, add_non_symmetric_;

  int m_, n_;
  std::string key_;
  BlockMatrix AT_;
  double lambda_;
  Eigen::MatrixXd B_;
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
