
#include "epsilon/algorithms/prox_admm.h"

#include <Eigen/SparseCholesky>

#include "epsilon/expression/expression.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/affine/affine.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_operator.h"
#include "epsilon/util/string.h"

ProxADMMSolver::ProxADMMSolver(
    const Problem& problem,
    const SolverParams& params,
    std::unique_ptr<ParameterService> parameter_service)
    : problem_(problem),
      params_(params),
      parameter_service_(std::move(parameter_service)) {}

void ProxADMMSolver::Init() {
  VLOG(2) << problem_.DebugString();
  var_map_.Insert(problem_);
  n_ = var_map_.n();

  // TODO(mwytock): Handle more general cases, 0 or >1 constraint
  CHECK_EQ(1, problem_.constraint_size());
  const Expression& constr = problem_.constraint(0);
  CHECK_EQ(Expression::INDICATOR, constr.expression_type());
  CHECK_EQ(Cone::ZERO, constr.cone().cone_type());

  // TODO(mwytock): This transform should be handled in a final pass by the
  // compiler.
  Expression expr;
  if (constr.arg_size() == 1) {
    expr = GetOnlyArg(constr);
  } else if (constr.arg_size() == 2) {
    expr = expression::Add(constr.arg(0), expression::Negate(constr.arg(1)));
  } else {
    LOG(FATAL) << "Constraint has more than 2 args\n" << constr.DebugString();
  }

  m_ = GetDimension(expr);
  {
    DynamicMatrix A_tmp(m_, n_);
    DynamicMatrix b_tmp(m_, 1);
    BuildAffineOperator(expr, var_map_, &A_tmp, &b_tmp);
    CHECK(A_tmp.is_sparse());
    A_ = A_tmp.sparse();
    b_ = b_tmp.AsDense();
  }

  VLOG(1) << "Prox ADMM, m = " << m_ << ", n = " << n_;

  CHECK_EQ(Expression::ADD, problem_.objective().expression_type());
  for (const Expression& expr : problem_.objective().arg()) {
    InitProxOperator(expr);
  }

  x_ = Eigen::VectorXd::Zero(n_);
  u_ = Eigen::VectorXd::Zero(m_);
  x_prev_ = Eigen::VectorXd::Zero(n_);
  x_param_prev_ = Eigen::VectorXd::Zero(n_);
  Ax_ = Eigen::VectorXd::Zero(m_);
  Ai_xi_norm_.resize(prox_ops_.size());
}

void ProxADMMSolver::InitProxOperator(const Expression& expr) {
  VLOG(2) << "InitProxOperator:\n" << expr.DebugString();

  // For now, we assume each prox function only operates on one variable but
  // this can be relaxed.
  VariableSet vars = GetVariables(expr);

  // TODO(mwytock): Should be pruned before getting here
  if (vars.size() == 0)
    return;

  ProxOperatorInfo prox;
  prox.var_map.Insert(expr);
  prox.V = GetProjection(var_map_, prox.var_map);
  prox.Ai = A_*prox.V.transpose();
  const int n = prox.var_map.n();

  VLOG(2) << "InitProxOperator, Ai:\n" << MatrixDebugString(prox.Ai);
  if (IsBlockScalar(prox.Ai)) {
    prox.B = SparseXd(n, m_);
    prox.linearized = false;

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
    for (SparseXd::InnerIterator iter(prox.Ai, 0); iter; ++iter) {
      CHECK(iter.value() != 0);

      const double alpha_i = iter.value();
      alpha_squared += alpha_i*alpha_i;
      prox.B.middleCols(iter.row(), n) = -alpha_i*SparseIdentity(n);
    }

    prox.B *= 1/alpha_squared;
    prox.op = CreateProxOperator(
        1/params_.rho()/alpha_squared, expr, prox.var_map);
  } else {
    prox.linearized = true;
    // TODO(mwytock): Figure out how to set this parameter
    prox.mu = 0.1;
    prox.op = CreateProxOperator(prox.mu, expr, prox.var_map);
  }

  prox.op->Init();
  prox.i = prox_ops_.size();
  prox_ops_.emplace_back(std::move(prox));
}

// TODO(mwytock): This involves a lot of matrix-vector products with Ai and V
// which are typically scalar identity matrices (by definition, in the case of
// applying the proximal operator directly). Find a way to optimize this better
// with more intelligence in the initialization phase.
void ProxADMMSolver::ApplyProxOperator(const ProxOperatorInfo& prox) {
  VLOG(2) << "ApplyProxOperator";

  const SparseXd& Ai = prox.Ai;
  const SparseXd& B = prox.B;
  const double mu = prox.mu;

  Eigen::VectorXd xi_old = prox.V*x_;
  Eigen::VectorXd xi;
  Eigen::VectorXd Ai_xi_old = Ai*xi_old;

  if (!prox.linearized) {
    xi = prox.op->Apply(B*(Ax_ - Ai_xi_old + b_ + u_));
  } else {
    xi = prox.op->Apply(xi_old - mu*params_.rho()*Ai.transpose()*(Ax_ + u_));
  }

  Eigen::VectorXd Ai_xi = Ai*xi;
  Ai_xi_norm_[prox.i] = Ai_xi.norm();
  x_ += prox.V.transpose()*(xi - xi_old);
  Ax_ += Ai_xi - Ai_xi_old;
}

void ProxADMMSolver::Solve() {
  Init();

  for (iter_ = 0; iter_ < params_.max_iterations(); iter_++) {
    x_prev_ = x_;
    for (const ProxOperatorInfo& op_info : prox_ops_)
      ApplyProxOperator(op_info);

    u_ += Ax_ + b_;

    VLOG(2) << "Iteration " << iter_ << "\n"
            << "x: " << VectorDebugString(x_) << "\n"
            << "u: " << VectorDebugString(u_);

    ComputeResiduals();
    LogStatus();
    UpdateStatus(status_);

    if (status_.state() == SolverStatus::OPTIMAL &&
        !params_.ignore_stopping_criteria()) {
      break;
    }

    if (HasExternalStop())
      break;
  }

  UpdateLocalParameters();
  if (iter_ == params_.max_iterations())
      status_.set_state(SolverStatus::MAX_ITERATIONS_REACHED);
  UpdateStatus(status_);
}

void ProxADMMSolver::UpdateLocalParameters() {
  for (const Expression* expr : GetVariables(problem_)) {
    const int i = var_map_.Get(expr->variable().variable_id());
    const int n = GetDimension(*expr);
    uint64_t param_id = VariableParameterId(
        problem_id(), expr->variable().variable_id());
    const Eigen::VectorXd delta = x_.segment(i, n) - x_param_prev_.segment(i, n);
    parameter_service_->Update(param_id, delta);
    x_param_prev_.segment(i, n) += delta;
  }
}

void ProxADMMSolver::ComputeResiduals() {
  SolverStatus::Residuals* r = status_.mutable_residuals();

  const double abs_tol = params_.abs_tol();
  const double rel_tol = params_.rel_tol();
  const double rho = params_.rho();

  double max_Ai_xi_norm = *std::max_element(
      Ai_xi_norm_.begin(), Ai_xi_norm_.end());

  // TODO(mwytock): Revisit this a bit more carefully, especially computation of
  // s_norm and epsilon_primal
  r->set_r_norm((Ax_ + b_).norm());
  r->set_s_norm((x_ - x_prev_).norm());
  r->set_epsilon_primal(abs_tol*sqrt(m_) + rel_tol*max_Ai_xi_norm);
  r->set_epsilon_dual(  abs_tol*sqrt(n_) + rel_tol*rho*(A_.transpose()*u_).norm());

  if (r->r_norm() <= r->epsilon_primal() &&
      r->s_norm() <= r->epsilon_dual()) {
    status_.set_state(SolverStatus::OPTIMAL);
  } else {
    status_.set_state(SolverStatus::RUNNING);
  }

  status_.set_num_iterations(iter_);
}

void ProxADMMSolver::LogStatus() {
  const SolverStatus::Residuals& r = status_.residuals();
  VLOG(1) << StringPrintf(
      "iter=%d residuals primal=%.2e [%.2e] dual=%.2e [%.2e]",
      status_.num_iterations(),
      r.r_norm(),
      r.epsilon_primal(),
      r.s_norm(),
      r.epsilon_dual());
}
