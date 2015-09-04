
#include "epsilon/algorithms/prox_admm.h"

#include <Eigen/SparseCholesky>
#include <gflags/gflags.h>

#include "epsilon/expression/expression.h"
#include "epsilon/expression/problem.h"
#include "epsilon/operators/affine.h"
#include "epsilon/operators/prox.h"
#include "epsilon/operators/vector_operator.h"
#include "epsilon/util/string.h"

// TODO(mwytock): Refactor into small class
bool RateLimitAllows(uint64_t limit, uint64_t* last) {
  uint64_t now = WallTime_Usec();
  if (now < *last + limit)
    return false;

  *last = now;
  return true;
}

ProxADMMSolver::ProxADMMSolver(
    const ProxProblem& problem,
    const SolverParams& params,
    std::unique_ptr<ParameterService> parameter_service)
    : problem_(problem),
      params_(params),
      parameter_service_(std::move(parameter_service)),
      last_consensus_usec_(0),
      last_status_usec_(0) {}

void ProxADMMSolver::Init() {
  VLOG(2) << problem_.DebugString();

  CHECK(problem_.prox_function_size() != 0);
  CHECK_EQ(problem_.equality_constraint_size(), 1);
  m_ = GetDimension(problem_.equality_constraint(0));

  n_ = 0;
  AddVariableOffsets(&problem_);
  for (const Expression* expr : GetVariables(problem_)) {
    var_map_[expr->variable().variable_id()] = expr;
    n_ += GetDimension(*expr);
  }

  VLOG(1) << "Consensus-Prox m = " << m_ << ", n = " << n_;
  {
    // Instantiate equality constraints
    DynamicMatrix A_tmp(m_, n_);
    DynamicMatrix b_tmp(m_, 1);
    BuildAffineOperator(problem_.equality_constraint(0), &A_tmp, &b_tmp);
    CHECK(A_tmp.is_sparse());
    A_ = A_tmp.sparse();
    b_ = b_tmp.AsDense();
  }

  for (const ProxFunction& f : problem_.prox_function()) {
    InitProxOperator(f);
  }
  for (const ConsensusVariable& cv : problem_.consensus_variable()) {
    InitConsensusVariable(cv);
  }

  x_ = Eigen::VectorXd::Zero(n_);
  u_ = Eigen::VectorXd::Zero(m_);
  x_prev_ = Eigen::VectorXd::Zero(n_);
  x_param_prev_ = Eigen::VectorXd::Zero(n_);
  Ax_ = Eigen::VectorXd::Zero(m_);
}

void ProxADMMSolver::InitProxOperator(const ProxFunction& f_orig) {
  // For now, we assume each prox function only operates on one variable but
  // this can be relaxed.
  std::vector<const Expression *> vars = GetVariables(f_orig);
  CHECK_EQ(1, vars.size());
  const int i = vars[0]->variable().offset();
  const int n = GetDimension(*vars[0]);

  // Recompute variable offsets local to this function
  ProxFunction f = f_orig;
  AddVariableOffsets(&f);

  ProxOperatorInfo info;
  info.i = i;
  info.n = n;

  SparseXd Ai = A_.middleCols(i, n);
  VLOG(2) << "InitProxOperator, Ai:\n" << MatrixDebugString(Ai);

  if (IsBlockScalar(Ai)) {
    info.B = SparseXd(n, m_);
    info.linearized = false;

    // Case in which variable shows up in the equality contraint only through
    // scalar multiplication. We can combine multiple terms through the
    // following logic:
    //
    // argmin_x f(x) + 1/2||a1*x - v1||^2 + 1/2||a2*x - v2||^2
    // is equivalent to:
    // argmin_x f(x) + (a1^2 + a2^2)/2||x - (a1v1 + a2v2)/(a1 + a2)^2||^2

    double alpha_squared = 0;
    // Iterate through first column in order to find the blocks where this
    // variable shows up in the constraints
    for (SparseXd::InnerIterator iter(Ai, 0); iter; ++iter) {
      CHECK(iter.value() != 0);

      const double alpha_i = iter.value();
      alpha_squared += alpha_i*alpha_i;
      info.B.middleCols(iter.row(), n) = -alpha_i*SparseIdentity(n);
    }

    info.B *= 1/alpha_squared;
    info.op = CreateProxOperator(f, 1/params_.rho()/alpha_squared, n);
  } else {
    info.linearized = true;
    // TODO(mwytock): Figure out how to set this parameter
    info.mu = 0.1;
    info.op = CreateProxOperator(f, info.mu, n);
  }

  info.op->Init();
  prox_ops_.emplace_back(std::move(info));
}

void ProxADMMSolver::ApplyProxOperator(const ProxOperatorInfo& prox) {
  const int i = prox.i;
  const int n = prox.n;
  const SparseXd& Ai = A_.middleCols(i, n);
  Eigen::VectorXd Ai_xi_old = Ai*x_.segment(i, n);

  if (!prox.linearized) {
    x_.segment(i, n) = prox.op->Apply(prox.B*(Ax_ - Ai_xi_old + b_ + u_));
  } else {
    x_.segment(i, n) = prox.op->Apply(
        x_.segment(i, n) - prox.mu*params_.rho()*Ai.transpose()*(Ax_ + u_));
  }

  Ax_ += Ai*x_.segment(i, n) - Ai_xi_old;
}

void ProxADMMSolver::Solve() {
  Init();

  for (iter_ = 0; iter_ < params_.max_iterations(); iter_++) {
    x_prev_ = x_;
    for (const ProxOperatorInfo& op_info : prox_ops_)
      ApplyProxOperator(op_info);

    if (RateLimitAllows(
            params_.consensus_rate_limit_usec(),
            &last_consensus_usec_)) {
      for (const ConsensusVariableInfo& var : consensus_vars_) {
        UpdateConsensusVariable(var);
      }
    }

    u_ += Ax_ + b_;

    VLOG(2) << "Iteration " << iter_ << "\n"
            << "x: " << VectorDebugString(x_) << "\n"
            << "u: " << VectorDebugString(u_);

    ComputeResiduals();
    LogStatus();

    if (RateLimitAllows(
            params_.status_rate_limit_usec(),
            &last_status_usec_)) {
      UpdateStatus(status_);
    }

    if (status_.state() == ProblemStatus::OPTIMAL &&
        !params_.ignore_stopping_criteria()) {
      break;
    }

    if (HasExternalStop())
      break;
  }

  UpdateLocalParameters();
  if (iter_ == params_.max_iterations())
      status_.set_state(ProblemStatus::MAX_ITERATIONS_REACHED);
  UpdateStatus(status_);
}

void ProxADMMSolver::InitConsensusVariable(const ConsensusVariable& cv) {
  ConsensusVariableInfo info;

  auto iter = var_map_.find(cv.variable_id());
  CHECK(iter != var_map_.end());
  const Expression* expr = iter->second;
  const int i = expr->variable().offset();
  const int n = GetDimension(*expr);
  info.i = i;
  info.n = n;
  info.B = SparseXd(n, m_);
  info.param_id = VariableId(problem_id(), expr->variable().variable_id());
  SparseXd Ai = A_.middleCols(i, n);
  CHECK(IsBlockScalar(Ai));

  for (SparseXd::InnerIterator iter(Ai, 0); iter; ++iter) {
    CHECK(iter.value() != 0);

    const double alpha = iter.value();
    info.B.middleCols(iter.row(), n) = -SparseIdentity(n)/alpha;
  }
  info.B *= 1./cv.num_instances();

  consensus_vars_.push_back(info);
  consensus_vars_set_.insert(cv.variable_id());
  VLOG(2) << "InitConsensusVariable " << cv.variable_id() << "\n"
          << "Ai:\n" << MatrixDebugString(Ai) << "\n"
          << "B:\n" << MatrixDebugString(info.B);
}

void ProxADMMSolver::UpdateConsensusVariable(
    const ConsensusVariableInfo& cv) {
  const int i = cv.i;
  const int n = cv.n;
  const SparseXd& Ai = A_.middleCols(i, n);
  Eigen::VectorXd Ai_xi_old = Ai*x_.segment(i, n);

  Eigen::VectorXd delta = cv.B*(Ax_ - Ai_xi_old + b_ + u_) -
                          x_param_prev_.segment(i, n);
  x_param_prev_.segment(i, n) += delta;
  x_.segment(i, n) = parameter_service_->Update(cv.param_id, delta);

  Ax_ += Ai*x_.segment(i, n) - Ai_xi_old;
}

void ProxADMMSolver::UpdateLocalParameters() {
  for (const auto& iter : var_map_) {
    if (consensus_vars_set_.find(iter.first) != consensus_vars_set_.end())
      continue;

    const Expression& expr = *iter.second;
    const int i = expr.variable().offset();
    const int n = GetDimension(expr);
    const Eigen::VectorXd delta = x_.segment(i, n) - x_param_prev_.segment(i, n);
    parameter_service_->Update(VariableId(problem_id(), iter.first), delta);
    x_param_prev_.segment(i, n) += delta;
  }
}

void ProxADMMSolver::ComputeResiduals() {
  ProblemStatus::Residuals* r = status_.mutable_residuals();

  const double abs_tol = params_.abs_tol();
  const double rel_tol = params_.rel_tol();
  const double rho = params_.rho();

  double max_Axi_norm = b_.norm();
  for (const ProxOperatorInfo& prox : prox_ops_) {
    const int i = prox.i;
    const int n = prox.n;
    const SparseXd& Ai = A_.middleCols(i, n);
    double Axi_norm = (Ai*x_.segment(i, n)).norm();
    if (Axi_norm > max_Axi_norm)
      max_Axi_norm = Axi_norm;
  }

  // TODO(mwytock): Revisit this a bit more carefully, especially computation of
  // s_norm and epsilon_primal
  r->set_r_norm((Ax_ + b_).norm());
  r->set_s_norm((x_ - x_prev_).norm());
  r->set_epsilon_primal(abs_tol*sqrt(m_) + rel_tol*max_Axi_norm);
  r->set_epsilon_dual(  abs_tol*sqrt(n_) + rel_tol*rho*(A_.transpose()*u_).norm());

  if (r->r_norm() <= r->epsilon_primal() &&
      r->s_norm() <= r->epsilon_dual()) {
    status_.set_state(ProblemStatus::OPTIMAL);
  } else {
    status_.set_state(ProblemStatus::RUNNING);
  }

  status_.set_num_iterations(iter_);
}

void ProxADMMSolver::LogStatus() {
  const ProblemStatus::Residuals& r = status_.residuals();
  VLOG(1) << StringPrintf(
      "iter=%d residuals primal=%.2e [%.2e] dual=%.2e [%.2e]",
      status_.num_iterations(),
      r.r_norm(),
      r.epsilon_primal(),
      r.s_norm(),
      r.epsilon_dual());
}
