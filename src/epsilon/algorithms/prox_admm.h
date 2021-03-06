#ifndef EPSILON_ALGORITHMS_CONSENSUS_PROX_H
#define EPSILON_ALGORITHMS_CONSENSUS_PROX_H

#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>

#include "epsilon/algorithms/solver.h"
#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/expression/var_offset_map.h"
#include "epsilon/prox/prox.h"
#include "epsilon/solver_params.pb.h"
#include "epsilon/vector/block_matrix.h"
#include "epsilon/vector/block_vector.h"
#include "epsilon/vector/vector_operator.h"
#include "epsilon/vector/vector_util.h"

class ProxADMMSolver final : public Solver {
public:
  ProxADMMSolver(
      const Problem& problem,
      const DataMap& data_map,
      const SolverParams& params);
  BlockVector Solve() override;

private:
  void Init();
  void InitConstraints();
  void InitProxOperators();
  void InitVariables();

  void ComputeResiduals();
  void LogStatus();
  BlockVector GetSolution();

  // Inputs
  Problem problem_;
  const DataMap& data_map_;
  SolverParams params_;

  bool initialized_;

  // Problem parameters
  int m_, n_, N_;
  BlockMatrix A_;
  BlockVector b_;
  std::vector<BlockMatrix> AiT_;
  std::vector<std::unique_ptr<ProxOperator> > prox_;

  // Iteration variables
  int iter_;
  BlockVector u_;
  std::vector<BlockVector> x_;
  std::vector<BlockVector> y_;

  // Iteration variables
  SolverStatus status_;

  // For computing residuals
  std::vector<BlockVector> y_prev_;
  BlockMatrix AT_;

  friend class ProxADMMSolverTest;
};


#endif  // EPSILON_ALGORITHMS_CONSENSUS_PROX_H
