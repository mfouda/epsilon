#include "epsilon/prox/prox.h"
#include "epsilon/prox/scaled_zone.h"

class NormL1AsymmetricProx final : public ScaledZoneProx {
public:
  NormL1AsymmetricProx() : ScaledZoneProx(0, 0, 0, 0) {};

  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    alpha_ = arg.f_expr().arg(0).arg(0).arg(0).constant().scalar();
    beta_ = arg.f_expr().arg(0).arg(1).arg(0).constant().scalar();
  }
};
REGISTER_PROX_OPERATOR(NormL1AsymmetricProx);

// Expression tree:
// (1, 1)    	  INDICATOR (cone: NON_NEGATIVE, prox: NormL1AsymmetricEpigraph)
// (1, 1)    	    ADD
// (1, 1)    	      VARIABLE (variable_id: cvxpy:1)
// (1, 1)    	      NEGATE
// (1, 1)    	        SUM
// (10, 1)   	          ADD
// (10, 1)   	            MULTIPLY
// (1, 1)    	              CONSTANT (scalar: 0.75)
// (10, 1)   	              MAX_ELEMENTWISE
// (10, 1)   	                VARIABLE (variable_id: cvxpy:0)
// (1, 1)    	                CONSTANT (scalar: 0)
// (10, 1)   	            MULTIPLY
// (1, 1)    	              CONSTANT (scalar: 0.25)
// (10, 1)   	              MAX_ELEMENTWISE
// (10, 1)   	                NEGATE
// (10, 1)   	                  VARIABLE (variable_id: cvxpy:0)
// (1, 1)    	                CONSTANT (scalar: 0)
class NormL1AsymmetricEpigraph final : public ScaledZoneEpigraph {
public:
  NormL1AsymmetricEpigraph() : ScaledZoneEpigraph(0, 0, 0, 0) {};

  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    alpha_ = arg.f_expr().arg(0).arg(1).arg(0).arg(0).arg(0).arg(0).constant().scalar();
    beta_ = arg.f_expr().arg(0).arg(1).arg(0).arg(0).arg(1).arg(0).constant().scalar();
  }
};
REGISTER_PROX_OPERATOR(NormL1AsymmetricEpigraph);
