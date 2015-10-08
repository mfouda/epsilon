
#include "epsilon/algorithms/prox_admm.h"

#include <Eigen/OrderingMethods>
#include <Eigen/SparseQR>

#include "epsilon/affine/affine.h"
#include "epsilon/affine/affine_matrix.h"
#include "epsilon/affine/split.h"
#include "epsilon/expression/expression.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/prox/prox.h"
#include "epsilon/util/string.h"
#include "epsilon/vector/vector_operator.h"
#include "epsilon/vector/vector_util.h"

// Handles constraints of the form AXB + C
class MatrixLeastSquaresOperator final : public VectorOperator {
 public:
  MatrixLeastSquaresOperator(
      const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    Vm_ = A.rows();
    Vn_ = B.cols();
    svd_A_.compute(A, Eigen::ComputeThinU|Eigen::ComputeThinV);
    svd_BT_.compute(B.transpose(), Eigen::ComputeThinU|Eigen::ComputeThinV);
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd& v) override {
    Eigen::MatrixXd XB = svd_A_.solve(ToMatrix(v, Vm_, Vn_));
    return ToVector(svd_BT_.solve(XB.transpose()).transpose());
  }

 private:
  int Vm_, Vn_;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd_A_;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd_BT_;
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
  std::vector<Expression> constr_args;
  for (const Expression& constr : problem_.constraint()) {
    CHECK_EQ(Expression::INDICATOR, constr.expression_type());
    CHECK_EQ(Cone::ZERO, constr.cone().cone_type());
    CHECK_EQ(1, constr.arg_size());

    ConstraintInfo info;
    info.mi = GetDimension(constr.arg(0));
    DynamicMatrix bi = DynamicMatrix::Zero(info.mi, 1);
    constr_args.push_back(expression::Reshape(constr.arg(0), info.mi, 1));

    VariableOffsetMap empty_var_map;
    SplitExpressionIterator iter(constr.arg(0));
    for (; !iter.done(); iter.NextValue()) {
      if (iter.leaf().expression_type() == Expression::VARIABLE) {
        info.exprs_by_var[iter.leaf().variable().variable_id()].push_back(
            iter.chain());
      } else {
        CHECK_EQ(iter.leaf().expression_type(), Expression::CONSTANT);
        BuildAffineOperator(iter.chain(), empty_var_map, nullptr, &bi);
      }
    }

    b_ = VStack(b_, bi.AsDense());
    constraints_.push_back(info);
  }
  m_ = b_.size();
  VLOG(2) << "b: " << VectorDebugString(b_);

  // TODO(mwytock): Get rid of this
  constr_expr_ = expression::VStack(constr_args);
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
  s_xi_.resize(ops_.size());
}

void ProxADMMSolver::InitLeastSquares(const Expression& var_expr) {
  const std::string& var_id = var_expr.variable().variable_id();
  VLOG(1) << "InitLeastSquares " << var_id << "\n"
          << var_expr.size().DebugString();
  VLOG(2) << "InitLeastSquares:\n" << var_expr.DebugString();

  OperatorInfo info;
  info.linearized = false;

  Eigen::MatrixXd A, B;
  int i = 0;
  int j = 0;
  std::vector<Eigen::Triplet<double> >  C_coeffs;
  for (const ConstraintInfo& constraint : constraints_) {
    auto iter = constraint.exprs_by_var.find(var_id);
    if (iter != constraint.exprs_by_var.end()) {

      affine::MatrixOperator op = affine::BuildMatrixOperator(
          expression::Add(iter->second));
      CHECK(op.C.isZero());

      VLOG(2) << expression::Add(iter->second).DebugString();
      VLOG(2) << "Matrix Op:\n"
              << "A:\n" << MatrixDebugString(op.A)
              << "B:\n" << MatrixDebugString(op.B);


      if (!A.rows() && !B.rows()) {
        A = op.A;
        B = op.B;
      } else {
        const bool A_equal = IsMatrixEqual(op.A, A);
        const bool B_equal = IsMatrixEqual(op.B, B);

        if (A_equal && B_equal) {
          // No change
          LOG(FATAL) << "Not implemented";
        } else if (A_equal) {
          B = HStack(B, op.B);
        } else if (B_equal) {
          A = VStack(A, op.A);
          LOG(FATAL) << "Not implemented";
        } else {
          LOG(FATAL) << "Incompatible matrix constraints\n"
                     << "A1: " << MatrixDebugString(A) << "\n"
                     << "B1: " << MatrixDebugString(B) << "\n"
                     << "A2: " << MatrixDebugString(op.A) << "\n"
                     << "B2: " << MatrixDebugString(op.B);
        }
      }

      AppendBlockTriplets(-SparseIdentity(constraint.mi), i, j, &C_coeffs);
      i += constraint.mi;
    }

    j += constraint.mi;
  }

  VLOG(1) << "InitLeastSquares, "
          << "A:\n" << MatrixDebugString(A) << "\n"
          << "B:\n" << MatrixDebugString(B);

  info.B = SparseXd(A.rows()*B.cols(), m_);
  info.B.setFromTriplets(C_coeffs.begin(), C_coeffs.end());
  info.op = std::unique_ptr<VectorOperator>(
      new MatrixLeastSquaresOperator(A, B));
  info.op->Init();

  VariableOffsetMap var_map;
  var_map.Insert(var_expr);
  info.Ai = DynamicMatrix::Zero(m_, var_map.n());
  BuildAffineOperator(constr_expr_, var_map, &info.Ai, nullptr);
  info.V = GetProjection(var_map_, var_map);
  info.i = ops_.size();
  ops_.emplace_back(std::move(info));
}

void ProxADMMSolver::InitProxOperator(const Expression& expr) {
  VLOG(2) << "InitProxOperator " << expr.DebugString();

  // TODO(mwytock): Should be pruned before getting here
  VariableSet vars = GetVariables(expr);
  if (vars.size() == 0)
    return;

  VariableOffsetMap var_map;
  OperatorInfo info;
  var_map.Insert(expr);
  const int n = var_map.n();

  info.V = GetProjection(var_map_, var_map);
  {
    // Build Ai
    std::vector<Eigen::Triplet<double> > Ai_coeffs;
    int i = 0;
    for (const ConstraintInfo& constraint : constraints_) {
      DynamicMatrix Aik = DynamicMatrix::Zero(constraint.mi, n);
      for (const Expression* var_expr : vars) {
        auto iter = constraint.exprs_by_var.find(var_expr->variable().variable_id());
        if (iter == constraint.exprs_by_var.end())
          continue;

        for (const Expression& expr : iter->second)
          BuildAffineOperator(expr, var_map, &Aik, nullptr);
      }
      CHECK(Aik.is_sparse());
      AppendBlockTriplets(Aik.sparse(), i, 0, &Ai_coeffs);
      i += constraint.mi;
    }
    info.Ai = DynamicMatrix::FromSparse(BuildSparseMatrix(m_, n, Ai_coeffs));
  }
  LOG(INFO) << "New:\n" << SparseMatrixDebugString(info.Ai.sparse());



  const SparseXd& Ai = info.Ai.sparse();
  SparseXd ATA = Ai.transpose()*Ai;
  VLOG(1) << "InitProxOperator " << expr.proximal_operator().name()
          << ", ATA:\n" << SparseMatrixDebugString(ATA);
  double alpha;
  if (IsScalarMatrix(ATA, &alpha)) {
    VLOG(1) << "Using standard ADMM\n";
    info.linearized = false;
    info.op = CreateProxOperator(1/params_.rho()/alpha, expr, var_map);
    info.B = -Ai.transpose()/alpha;
  } else {
    VLOG(1) << "Using linearized ADMM";
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
  const double mu = info.mu;

  Eigen::VectorXd xi_old = info.V*x_;
  Eigen::VectorXd xi;
  Eigen::VectorXd Ai_xi_old = Ai.Apply(xi_old);

  if (!info.linearized) {
    xi = info.op->Apply(info.B*(Ax_ - Ai_xi_old + b_ + u_));
  } else {
    xi = info.op->Apply(xi_old - mu*params_.rho()*Ai.ApplyTranspose((Ax_ + u_)));
  }

  Eigen::VectorXd Ai_xi = Ai.Apply(xi);
  Ai_xi_norm_[info.i] = Ai_xi.norm();
  s_xi_[info.i] = Ai.ApplyTranspose(Ax_);
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

  double Ai_xi_norm_inf = fmax(
      *std::max_element(Ai_xi_norm_.begin(), Ai_xi_norm_.end()),
      b_.norm());
  double ATu_norm_squared = 0.0;
  double s_norm_squared = 0.0;

  for (int i = 0; i < ops_.size(); i++) {
    const DynamicMatrix& Ai = ops_[i].Ai;
    ATu_norm_squared += Ai.ApplyTranspose(u_).squaredNorm();
    s_norm_squared += (Ai.ApplyTranspose(Ax_) - s_xi_[i]).squaredNorm();
  }

  r->set_r_norm((Ax_ + b_).norm());
  r->set_s_norm(rho*sqrt(s_norm_squared));
  r->set_epsilon_primal(abs_tol*sqrt(m_) + rel_tol*Ai_xi_norm_inf);
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
