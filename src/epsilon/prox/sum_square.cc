
#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"
#include "epsilon/vector/block_cholesky.h"
#include "epsilon/prox/newton.h"

// ||H(x)||_2^2
class SumSquareProx final : public ProxOperator {
 public:
  void Init(const ProxOperatorArg& arg) override {
    const BlockMatrix& H = arg.affine_arg().A;
    const BlockVector& g = arg.affine_arg().b;
    const BlockMatrix& A = arg.affine_constraint().A;
    const double alpha = sqrt(2*arg.prox_function().alpha());

    // [ 0   H'  A'][ x ] = [ 0 ]
    // [ H  -I   0 ][ y ]   [-g ]
    // [ A   0  -I ][ z ]   [ v ]
    // TODO(mwytock): Cholesky factorization should not require full matrix
    // since it is symmetric.
    BlockMatrix M = alpha*(H + H.Transpose()) + (A + A.Transpose())
                    - H.LeftIdentity() - A.LeftIdentity();
    VLOG(2) << "M: " << M.DebugString();
    chol_.Compute(M);
    b_ = -alpha*g;
    var_keys_ = H.col_keys();
  }

  BlockVector Apply(const BlockVector& v) override {
    return chol_.Solve(b_ + v).Select(var_keys_);
  }

 private:
  BlockCholesky chol_;
  BlockVector b_;
  std::set<std::string> var_keys_;
};
REGISTER_PROX_OPERATOR(SUM_SQUARE, SumSquareProx);

class SumSquareEpigraph final : public VectorProx {
  void ApplyVector(
      const VectorProxInput& input,
      VectorProxOutput* output) {
    // solve |u|^2 - (lam+s)(1+2*lam)^2 = 0
    const Eigen::VectorXd& u = input.value_vec(0);
    const double s =  input.value(1);
    double lam = LargestRealCubicRoot(1+s, 0.25+s, (s-u.squaredNorm())/4);
    if(lam < 0)
      lam = 0;
    Eigen::VectorXd x = u/(1+2*lam);
  output->set_value(0, x);
  output->set_value(1, s+lam);
  }
};
REGISTER_EPIGRAPH_OPERATOR(SUM_SQUARE, SumSquareEpigraph);
