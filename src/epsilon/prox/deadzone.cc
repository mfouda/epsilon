#include "epsilon/prox/scaled_zone.h"

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_util.h"

class DeadZoneProx final : public ScaledZoneProx {
public:
  DeadZoneProx() : ScaledZoneProx(1, 1, 0, 0) {};

  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    M_ = arg.f_expr().arg(0).arg(0).arg(0).arg(1).arg(0).constant().scalar();
    VLOG(1) << "Deadzone Prox : M = " << M_ << "\n";
  }
};
REGISTER_PROX_OPERATOR(DeadZoneProx);

// (1, 1)    	  INDICATOR (cone: NON_NEGATIVE, prox: DeadZoneEpigraph)
// (1, 1)    	    ADD
// (1, 1)    	      VARIABLE (variable_id: cvxpy:1)
// (1, 1)    	      NEGATE
// (1, 1)    	        SUM
// (10, 1)   	          ADD
// (10, 1)   	            MAX_ELEMENTWISE
// (10, 1)   	              ADD
// (10, 1)   	                VARIABLE (variable_id: cvxpy:0)
// (1, 1)    	                NEGATE
// (1, 1)    	                  CONSTANT (scalar: 1)
// (1, 1)    	              CONSTANT (scalar: 0)
// (10, 1)   	            MAX_ELEMENTWISE
// (10, 1)   	              ADD
// (10, 1)   	                NEGATE
// (10, 1)   	                  VARIABLE (variable_id: cvxpy:0)
// (1, 1)    	                NEGATE
// (1, 1)    	                  CONSTANT (scalar: 1)
// (1, 1)    	              CONSTANT (scalar: 0)
class DeadZoneEpigraph final : public ScaledZoneEpigraph {
public:
  DeadZoneEpigraph() : ScaledZoneEpigraph(1, 1, 0, 0) {};

  void Init(const ProxOperatorArg& arg) override {
    lambda_ = arg.lambda();
    M_ = arg.f_expr().arg(0).arg(1).arg(0).arg(0).arg(0).arg(0).arg(1).arg(0).constant().scalar();
    VLOG(1) << "Deadzone Epigraph : M = " << M_ << "\n";
  }
};
REGISTER_PROX_OPERATOR(DeadZoneEpigraph);
