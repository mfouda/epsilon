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
#include "epsilon/vector/dynamic_matrix.h"
#include "epsilon/vector/vector_operator.h"
#include "epsilon/vector/vector_util.h"

class ProxADMMSolver final : public Solver {
public:
  ProxADMMSolver(
      const Problem& problem,
      const SolverParams& params,
      std::unique_ptr<ParameterService> parameter_service);
  void Solve() override;

private:
  void Init();
  void InitVariables();
  void InitConstraints();
  std::unique_ptr<BlockVectorOperator> InitProxOperator(
      const Expression& expr);

  void ApplyProxOperator(int i);
  void ComputeResiduals();
  void LogStatus();
  void UpdateLocalParameters();

  // Inputs
  Problem problem_;
  SolverParams params_;

  // Stores parameter
  std::unique_ptr<ParameterService> parameter_service_;

  // Problem parameters
  int m_, n_;
  BlockMatrix A_;
  BlockVector b_;

  // Iteration variables
  int iter_;
  BlockVector x_, x_prev_, u_, v_;
  SolverStatus status_;

  // For computing residuals
  std::vector<double> Ai_xi_norm_;
  std::vector<Eigen::VectorXd> s_;

  // Precomputed
  std::vector<std::unique_ptr<BlockVectorOperator>> prox_;
};


#endif  // EPSILON_ALGORITHMS_CONSENSUS_PROX_H
