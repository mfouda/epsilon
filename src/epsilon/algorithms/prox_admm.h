#ifndef EPSILON_ALGORITHMS_CONSENSUS_PROX_H
#define EPSILON_ALGORITHMS_CONSENSUS_PROX_H

#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>

#include "epsilon/algorithms/solver.h"
#include "epsilon/operators/vector_operator.h"
#include "epsilon/parameters/parameter_service.h"
#include "epsilon/prox.pb.h"
#include "epsilon/solver_params.pb.h"
#include "epsilon/util/vector.h"

struct ProxOperatorInfo {
  bool linearized;
  int i, n;
  std::unique_ptr<VectorOperator> op;

  // Maps R^m -> R^ni for input to prox operator
  SparseXd B;

  double mu;
};

struct ConsensusVariableInfo {
  int i, n;

  uint64_t param_id;

  // Maps R^m -> R^ni for consensus variable update
  SparseXd B;
};

class ProxADMMSolver final : public Solver {
public:
  ProxADMMSolver(
      const ProxProblem& problem,
      const SolverParams& params,
      std::unique_ptr<ParameterService> parameter_service);
  void Solve() override;

private:
  void Init();
  void InitProxOperator(const ProxFunction& f);
  void InitConsensusVariable(const ConsensusVariable& cv);
  void ApplyProxOperator(const ProxOperatorInfo& op);
  void UpdateConsensusVariable(const ConsensusVariableInfo& cv);
  void UpdateLocalParameters();
  void ComputeResiduals();
  void LogStatus();

  // Inputs
  ProxProblem problem_;
  SolverParams params_;

  // Stores parameter
  std::unique_ptr<ParameterService> parameter_service_;

  // Problem size and number of prox functions
  int m_, n_;

  // Iteration variables
  int iter_;
  Eigen::VectorXd x_, x_prev_, x_param_prev_, u_, Ax_;
  ProblemStatus status_;

  // Equality constraints
  SparseXd A_;
  Eigen::VectorXd b_;

  // Precomputed
  std::unordered_map<std::string, const Expression*> var_map_;
  std::unordered_set<std::string> consensus_vars_set_;
  std::vector<ProxOperatorInfo> prox_ops_;
  std::vector<ConsensusVariableInfo> consensus_vars_;

  // Keeps tracking of consensus updates
  uint64_t last_consensus_usec_;
  uint64_t last_status_usec_;
};


#endif  // EPSILON_ALGORITHMS_CONSENSUS_PROX_H
