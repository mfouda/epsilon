#ifndef EPSILON_ALGORITHMS_CONSENSUS_PROX_H
#define EPSILON_ALGORITHMS_CONSENSUS_PROX_H

#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>

#include "epsilon/algorithms/solver.h"
#include "epsilon/expression.pb.h"
#include "epsilon/expression/var_offset_map.h"
#include "epsilon/vector/vector_operator.h"
#include "epsilon/parameters/parameter_service.h"
#include "epsilon/solver_params.pb.h"
#include "epsilon/vector/vector_util.h"

struct ProxOperatorInfo {
  int i;
  std::unique_ptr<VectorOperator> op;

  // Holds the block of the constraint matrix
  SparseXd Ai;

  // Maps R^m -> R^ni for input to prox operator.
  SparseXd B;

  // Maps R^n -> R^ni for output of prox operator
  SparseXd V;

  // For linearization
  bool linearized;
  double mu;

  VariableOffsetMap var_map;
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
  void InitProxOperator(const Expression& expr);
  void ApplyProxOperator(const ProxOperatorInfo& op);
  void ComputeResiduals();
  void LogStatus();
  void UpdateLocalParameters();

  // Inputs
  Problem problem_;
  SolverParams params_;

  // Stores parameter
  std::unique_ptr<ParameterService> parameter_service_;

  // Problem size and number of prox functions
  int m_, n_;

  // Iteration variables
  int iter_;
  Eigen::VectorXd x_, x_prev_, x_param_prev_, u_, Ax_;
  SolverStatus status_;
  std::vector<double> Ai_xi_norm_;

  // Equality constraints
  SparseXd A_;
  Eigen::VectorXd b_;

  // Precomputed
  VariableOffsetMap var_map_;
  std::vector<ProxOperatorInfo> prox_ops_;
};


#endif  // EPSILON_ALGORITHMS_CONSENSUS_PROX_H
