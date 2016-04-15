#ifndef EPSILON_ALGORITHMS_PROX_ADMM_TWO_BLOCK_H
#define EPSILON_ALGORITHMS_PROX_ADMM_TWO_BLOCK_H

#include "epsilon/algorithms/solver.h"
#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/expression/var_offset_map.h"
#include "epsilon/parameters/parameter_service.h"
#include "epsilon/prox/prox.h"
#include "epsilon/solver_params.pb.h"
#include "epsilon/vector/block_matrix.h"
#include "epsilon/vector/block_vector.h"
#include "epsilon/vector/vector_operator.h"
#include "epsilon/vector/vector_util.h"

// This solver applies ADMM in a two block fashion. In particular, given a
// problem of the standard Epsilon form:
//
// minimize    f_1(x_1) + ... + f_N(x_N)
// subject to  A_1*x_1  + ... + A_N*x_N = b
//
// we introduce a copy of the variables (z_1, ... z_N) and apply the updates:
//
// x_i = argmin_xi f_i(x_i)  + (rho/2)||x_i - z_i + u_i||_2^2
// z   = argmin_z  I(Az = b) + (rho/2)||x - z + u||_2^2
// u   = u + x - z
class ProxADMMTwoBlockSolver final : public Solver {
public:
  ProxADMMTwoBlockSolver(
      const Problem& problem,
      const SolverParams& params);
  BlockVector Solve() override;

private:
  void Init();
  void InitConstraints();
  void InitProxOperators();

  void ComputeResiduals();
  void LogStatus();

  // Inputs
  SolverParams params_;

  // Stores parameter
  std::unique_ptr<ParameterService> parameter_service_;

  // Problem parameters
  int m_, n_, N_;

  // Problem data
  std::vector<std::unique_ptr<ProxOperator>> prox_;
  std::unique_ptr<ProxOperator> constr_prox_;

  // Iteration variables
  int iter_;
  BlockVector x_;
  BlockVector z_, z_prev_;
  BlockVector u_;

  // Iteration variables
  SolverStatus status_;

  // For computing residuals
  std::vector<BlockVector> y_prev_;
  BlockMatrix AT_;
};


#endif  // EPSILON_ALGORITHMS_PROX_ADMM_TWO_BLOCK_H
