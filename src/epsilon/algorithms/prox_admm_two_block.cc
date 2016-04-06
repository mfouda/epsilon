
#include "epsilon/algorithms/prox_admm_two_block.h"

#include "epsilon/affine/affine.h"
#include "epsilon/expression/expression.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/util/logging.h"
#include "epsilon/util/string.h"
#include "epsilon/vector/vector_operator.h"
#include "epsilon/vector/vector_util.h"

ProxADMMTwoBlockSolver::ProxADMMTwoBlockSolver(
    const Problem& problem,
    const SolverParams& params,
    std::unique_ptr<ParameterService> parameter_service)
    : problem_(problem),
      params_(params),
      parameter_service_(std::move(parameter_service)) {}

void ProxADMMTwoBlockSolver::InitConstraints() {
  const double sqrt_rho = sqrt(params_.rho());

  AffineOperator H, A;
  for (int i = 0; i < problem_.constraint_size(); i++) {
    const Expression& constr = problem_.constraint(i);
    CHECK_EQ(Expression::INDICATOR, constr.expression_type());
    CHECK_EQ(Cone::ZERO, constr.cone().cone_type());
    CHECK_EQ(1, constr.arg_size());
    affine::BuildAffineOperator(
        problem_.constraint(i).arg(0),
        affine::constraint_key(i),
        &H.A, &H.b);

    for (const Expression* expr : GetVariables(constr)) {
      const std::string& var_id = expr->variable().variable_id();
      A.A(var_id, var_id) =
          sqrt_rho*linear_map::Identity(GetDimension(*expr));
      z_(var_id) = BlockVector::DenseVector::Zero(GetDimension(*expr));
    }
  }
  // Prox for I(Ax + b = 0) constraint
  constr_prox_ = CreateProxOperator(ProxFunction::ZERO, false);
  constr_prox_->Init(ProxOperatorArg(ProxFunction(), H, A));
  VLOG(1) << "constr prox init done";

  m_ = H.A.m();
  n_ = H.A.n();
}

void ProxADMMTwoBlockSolver::InitProxOperators() {
  CHECK_EQ(Expression::ADD, problem_.objective().expression_type());
  N_ = problem_.objective().arg_size();

  const double sqrt_rho = sqrt(params_.rho());
  for (int i = 0; i < N_; i++) {
    const Expression& f_expr = problem_.objective().arg(i);

    VLOG(1) << "prox " << i << " build affine operator";
    AffineOperator H;
    for (int i = 0; i < f_expr.arg_size(); i++) {
      affine::BuildAffineOperator(
          f_expr.arg(i),
          affine::arg_key(i),
          &H.A, &H.b);
    }

    AffineOperator A;
    for (const Expression* expr : GetVariables(f_expr)) {
      const std::string& var_id = expr->variable().variable_id();
      A.A(var_id, var_id) =
          sqrt_rho*linear_map::Identity(GetDimension(*expr));
    }

    ProxFunction::Type type = f_expr.prox_function().prox_function_type();
    bool epigraph = f_expr.prox_function().epigraph();
    VLOG(1) << "prox " << i << ", initializing "
            << ProxFunction::Type_Name(type);
    prox_.emplace_back(CreateProxOperator(type, epigraph));
    prox_.back()->Init(ProxOperatorArg(f_expr.prox_function(), H, A));
    VLOG(1) << "prox " << i << " init done";
  }
}

void ProxADMMTwoBlockSolver::Init() {
  VLOG(3) << problem_.DebugString();
  InitConstraints();
  InitProxOperators();

  VLOG(1) << "Prox ADMM (two block), m = " << m_ << ", n = " << n_
          << ", N = " << N_;
}

void ProxADMMTwoBlockSolver::Solve() {
  Init();

  for (iter_ = 0; iter_ < params_.max_iterations(); iter_++) {
    z_prev_ = z_;

    BlockVector zu = z_ - u_;
    x_ = BlockVector();
    for (int i = 0; i < N_; i++) {
      x_ += prox_[i]->Apply(zu);
      VLOG(2) << "x[" << i << "]: " << x_.DebugString();
    }
    z_ = constr_prox_->Apply(x_ + u_);
    VLOG(2) << "z: " << z_.DebugString();

    u_ += x_ - z_;
    VLOG(2) << "u: " << u_.DebugString();

    if (iter_ % params_.epoch_iterations() == 0) {
      ComputeResiduals();
      if (status_.state() == SolverStatus::OPTIMAL)
        break;
    }

    if (iter_ % params_.log_iterations() == 0) {
      LogStatus();
    }
  }

  if (iter_ == params_.max_iterations()) {
    ComputeResiduals();
    status_.set_state(SolverStatus::MAX_ITERATIONS_REACHED);
  }

  LogStatus();
  UpdateParameters();
  UpdateStatus(status_);
}

void ProxADMMTwoBlockSolver::UpdateParameters() {
  for (int i = 0; i < N_; i++) {
    for (const Expression* expr : GetVariables(problem_.objective().arg(i))) {
      const std::string& var_id = expr->variable().variable_id();
      uint64_t param_id = VariableParameterId(problem_id(), var_id);
      parameter_service_->Update(param_id, x_(var_id));
    }
  }
}

void ProxADMMTwoBlockSolver::ComputeResiduals() {
  SolverStatus::Residuals* r = status_.mutable_residuals();

  const double abs_tol = params_.abs_tol();
  const double rel_tol = params_.rel_tol();
  const double rho = params_.rho();

  VLOG(3) << "set residuals";
  r->set_r_norm((x_ - z_).norm());
  r->set_s_norm(rho*(z_ - z_prev_).norm());
  r->set_epsilon_primal(abs_tol*sqrt(n_) + rel_tol*fmax(x_.norm(), z_.norm()));
  r->set_epsilon_dual(  abs_tol*sqrt(n_) + rel_tol*rho*u_.norm());

  if (r->r_norm() <= r->epsilon_primal() &&
      r->s_norm() <= r->epsilon_dual()) {
    status_.set_state(SolverStatus::OPTIMAL);
  } else {
    status_.set_state(SolverStatus::RUNNING);
  }

  status_.set_num_iterations(iter_);
}

void ProxADMMTwoBlockSolver::LogStatus() {
  const SolverStatus::Residuals& r = status_.residuals();
  std::string status = StringPrintf(
      "iter=%d residuals primal=%.2e [%.2e] dual=%.2e [%.2e]",
      status_.num_iterations(),
      r.r_norm(),
      r.epsilon_primal(),
      r.s_norm(),
      r.epsilon_dual());
  VLOG(1) << status;
  if (params_.verbose()) LogVerbose(status);
}
