#ifndef EPSILON_ALGORITHMS_CONSENSUS_PROX_H
#define EPSILON_ALGORITHMS_CONSENSUS_PROX_H

#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>

#include "epsilon/algorithms/solver.h"
#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/expression/var_offset_map.h"
#include "epsilon/parameters/parameter_service.h"
#include "epsilon/solver_params.pb.h"
#include "epsilon/vector/block_matrix.h"
#include "epsilon/vector/block_vector.h"
#include "epsilon/vector/vector_operator.h"
#include "epsilon/vector/vector_util.h"

struct ProxOperatorInfo {
  std::unique_ptr<BlockVectorOperator> op;
  VariableSet vars;
};

class ProxADMMSolver final : public Solver {
public:
  ProxADMMSolver(
      const Problem& problem,
      const SolverParams& params,
      std::unique_ptr<ParameterService> parameter_service);
  void Solve() override;

private:
  void Init();
  void InitConstraints();
  void InitProxOperators();

  void ComputeResiduals();
  void LogStatus();
  void UpdateLocalParameters();

  // Inputs
  Problem problem_;
  SolverParams params_;

  // Stores parameter
  std::unique_ptr<ParameterService> parameter_service_;

  // Problem parameters
  int m_, n_, N_;
  BlockMatrix A_;
  BlockVector b_;
  // (Ai)^T
  std::vector<BlockMatrix> AT_;

  // Iteration variables
  int iter_;
  BlockVector u_;
  std::vector<BlockVector> x_;
  std::vector<std::unique_ptr<BlockVectorOperator> > prox_;

  // Iteration variables
  SolverStatus status_;

  // For computing residuals
  std::vector<BlockVector> x_prev_;

  // Precomputed
};


#endif  // EPSILON_ALGORITHMS_CONSENSUS_PROX_H
