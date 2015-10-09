#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_util.h"
#include <cmath>

class OrthoInvariantProx: public ProxOperator {
public:
  virtual void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    ProxOperatorArg prox_arg(lambda_, NULL, NULL);
    f_->Init(prox_arg);
  }
  OrthoInvariantProx(std::unique_ptr<ProxOperator> f, bool to_symm=false)
    : f_(std::move(f)), to_symm_(to_symm) {}
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& y) override;
protected:
  std::unique_ptr<ProxOperator> f_;
  double lambda_;
  bool to_symm_;
};

class OrthoInvariantEpigraph: public ProxOperator {
public:
  virtual void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    ProxOperatorArg prox_arg(lambda_, NULL, NULL);
    f_->Init(prox_arg);
  }
  OrthoInvariantEpigraph(std::unique_ptr<ProxOperator> f, bool to_symm=false)
    : f_(std::move(f)), to_symm_(to_symm) {}
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& sy) override;

protected:
  std::unique_ptr<ProxOperator> f_;
  double lambda_;
  bool to_symm_;
};
