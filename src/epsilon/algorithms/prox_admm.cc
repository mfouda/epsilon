
#include "epsilon/algorithms/prox_admm.h"

#include <Eigen/SparseCholesky>

#include "epsilon/expression/expression.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/affine/affine.h"
#include "epsilon/prox/prox.h"
#include "epsilon/vector/vector_operator.h"
#include "epsilon/util/string.h"

class DenseLeastSquaresOperator final : public VectorOperator {
 public:
  DenseLeastSquaresOperator(const Eigen::MatrixXd& A) : A_(A) {}

  void Init() override {
    CHECK(A_.rows() >= A_.cols());
    ATA_solver_.compute(A_.transpose()*A_);
    CHECK_EQ(ATA_solver_.info(), Eigen::Success);
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    return ATA_solver_.solve(A_.transpose()*v);
  }

 private:
  Eigen::MatrixXd A_;
  Eigen::LLT<Eigen::MatrixXd> ATA_solver_;
};

ProxADMMSolver::ProxADMMSolver(
    const Problem& problem,
    const SolverParams& params,
    std::unique_ptr<ParameterService> parameter_service)
    : problem_(problem),
      params_(params),
      parameter_service_(std::move(parameter_service)) {}

void ProxADMMSolver::InitVariables() {
  var_map_.Insert(problem_);
  n_ = var_map_.n();
}

// TODO(mwytock): Dealing with constraints in the fashion: w/ vstack() and
// multiple calls to BuildAffineOperator() can be highly inefficient. We need a
// better mechanism, likely built on a more flexible BuildAffineOperator()
// implementation.
void ProxADMMSolver::InitConstraints() {
  std::vector<Expression> args;
  for (const Expression& constr : problem_.constraint()) {
    CHECK_EQ(Expression::INDICATOR, constr.expression_type());
    CHECK_EQ(Cone::ZERO, constr.cone().cone_type());
    CHECK_EQ(1, constr.arg_size());
    args.push_back(
        expression::Reshape(constr.arg(0), GetDimension(constr.arg(0)), 1));
  }

  // Get just the constant term
  constr_expr_ = expression::VStack(args);
  m_ = GetDimension(constr_expr_);
  DynamicMatrix b(m_, 1);
  VariableOffsetMap empty_var_map;
  BuildAffineOperator(constr_expr_, empty_var_map, nullptr, &b);
  b_ = b.AsDense();

  VLOG(2) << "b: " << VectorDebugString(b_);
}

void ProxADMMSolver::Init() {
  VLOG(2) << problem_.DebugString();
  InitVariables();
  InitConstraints();
  VLOG(1) << "Prox ADMM, m = " << m_ << ", n = " << n_;

  CHECK_EQ(Expression::ADD, problem_.objective().expression_type());
  for (const Expression& expr : problem_.objective().arg()) {
    InitProxOperator(expr);
  }

  VariableSet vars = GetVariables(problem_);
  for (const Expression* expr : vars) {
    if (vars_in_prox_.find(expr) == vars_in_prox_.end())
      InitLeastSquares(*expr);
  }

  x_ = Eigen::VectorXd::Zero(n_);
  u_ = Eigen::VectorXd::Zero(m_);
  x_prev_ = Eigen::VectorXd::Zero(n_);
  x_param_prev_ = Eigen::VectorXd::Zero(n_);
  Ax_ = Eigen::VectorXd::Zero(m_);
  Ai_xi_norm_.resize(ops_.size());
}

void ProxADMMSolver::InitLeastSquares(const Expression& var_expr) {
  VLOG(2) << "InitLeastSquares:\n" << var_expr.DebugString();

  VariableOffsetMap var_map;
  var_map.Insert(var_expr);
  OperatorInfo info;

  // Build Ai matrix, force dense
  info.Ai = DynamicMatrix::FromDense(Eigen::MatrixXd::Zero(m_, var_map.n()));
  BuildAffineOperator(constr_expr_, var_map, &info.Ai, nullptr);
  CHECK(!info.Ai.is_sparse());
  info.op = std::unique_ptr<VectorOperator>(
      new DenseLeastSquaresOperator(info.Ai.dense()));

  info.linearized = false;
  info.B = -SparseIdentity(m_);
  info.V = GetProjection(var_map_, var_map);
  info.op->Init();
  info.i = ops_.size();
  ops_.emplace_back(std::move(info));
}

void ProxADMMSolver::InitProxOperator(const Expression& expr) {
  VLOG(2) << "InitProxOperator:\n" << expr.DebugString();

  // TODO(mwytock): Should be pruned before getting here
  VariableSet vars = GetVariables(expr);
  if (vars.size() == 0)
    return;

  VariableOffsetMap var_map;
  OperatorInfo info;
  var_map.Insert(expr);
  const int n = var_map.n();

  info.V = GetProjection(var_map_, var_map);
  info.Ai = DynamicMatrix::Zero(m_, n);
  BuildAffineOperator(constr_expr_, var_map, &info.Ai, nullptr);
  CHECK(info.Ai.is_sparse());
  const SparseXd& Ai = info.Ai.sparse();

  VLOG(2) << "InitProxOperator, Ai:\n" << SparseMatrixDebugString(Ai);
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
    info.op = CreateProxOperator(
        1/params_.rho()/alpha_squared, expr, var_map);
  } else {
    VLOG(2) << "Using linearized ADMM";

    info.linearized = true;
    // TODO(mwytock): Figure out how to set this parameter
    info.mu = 0.1;
    info.op = CreateProxOperator(info.mu, expr, var_map);
  }

  info.op->Init();
  info.i = ops_.size();
  ops_.emplace_back(std::move(info));
  vars_in_prox_.insert(vars.begin(), vars.end());
}

// TODO(mwytock): This involves a lot of matrix-vector products with Ai and V
// which are typically scalar identity matrices (by definition, in the case of
// applying the proximal operator directly). Find a way to optimize this better
// with more intelligence in the initialization phase.
void ProxADMMSolver::ApplyOperator(const OperatorInfo& info) {
  VLOG(2) << "ApplyOperator";

  const DynamicMatrix& Ai = info.Ai;
  const SparseXd& B = info.B;
  const double mu = info.mu;

  Eigen::VectorXd xi_old = info.V*x_;
  Eigen::VectorXd xi;
  Eigen::VectorXd Ai_xi_old = Ai.Apply(xi_old);

  if (!info.linearized) {
    xi = info.op->Apply(B*(Ax_ - Ai_xi_old + b_ + u_));
  } else {
    xi = info.op->Apply(xi_old - mu*params_.rho()*Ai.ApplyTranspose((Ax_ + u_)));
  }

  Eigen::VectorXd Ai_xi = Ai.Apply(xi);
  Ai_xi_norm_[info.i] = Ai_xi.norm();
  x_ += info.V.transpose()*(xi - xi_old);
  Ax_ += Ai_xi - Ai_xi_old;

  VLOG(2) << "xi_old: " << VectorDebugString(xi_old);
  VLOG(2) << "xi: " << VectorDebugString(xi);
}

void ProxADMMSolver::Solve() {
  Init();

  for (iter_ = 0; iter_ < params_.max_iterations(); iter_++) {
    x_prev_ = x_;
    for (const OperatorInfo& info : ops_)
      ApplyOperator(info);

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

  double ATu_norm_squared = 0.0;
  for (const OperatorInfo& info : ops_)
    ATu_norm_squared += info.Ai.ApplyTranspose(u_).squaredNorm();

  // TODO(mwytock): Revisit this a bit more carefully, especially computation of
  // s_norm and epsilon_primal
  r->set_r_norm((Ax_ + b_).norm());
  r->set_s_norm((x_ - x_prev_).norm());
  r->set_epsilon_primal(abs_tol*sqrt(m_) + rel_tol*max_Ai_xi_norm);
  r->set_epsilon_dual(  abs_tol*sqrt(n_) + rel_tol*rho*(sqrt(ATu_norm_squared)));

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
