
#include "epsilon/algorithms/solver.h"

#include "epsilon/util/string.h"

class StatImpl final : public Stat {
 public:
  StatImpl(const std::string& name, const Timer* timer) : timer_(timer) {
    series_.set_name(name);
  }

  void AddValue(double value) {
    std::lock_guard<std::mutex> l(lock_);
    series_.add_time_usec(timer_->GetTimeUsec());
    series_.add_value(value);
  }

  void Fill(SolverStatSeries* series) {
    std::lock_guard<std::mutex> l(lock_);
    *series = series_;
  }

private:
  const Timer* timer_;  // Not owned

  std::mutex lock_;
  SolverStatSeries series_;
};

Solver::Solver(Problem problem) :
    problem_(std::move(problem)),
    problem_id_(std::hash<uint64_t>()(WallTime_Usec())) {
  InitParameterMap(problem_.mutable_objective());
  for (Expression& constr : *problem_.mutable_constraint())
    InitParameterMap(&constr);
}

void Solver::InitParameterMap(Expression* expr) {
  if (expr->expression_type() == Expression::CONSTANT &&
      expr->constant().parameter_id() != "") {
    parameter_map_[expr->constant().parameter_id()].push_back(
        expr->mutable_constant());
  }

  if (expr->expression_type() == Expression::LINEAR_MAP) {
    InitParameterMap(expr->mutable_linear_map());
  }

  for (Expression& arg : *expr->mutable_arg())
    InitParameterMap(&arg);
}

void Solver::InitParameterMap(LinearMap* linear_map) {
  if (linear_map->constant().parameter_id() != "") {
    parameter_map_[linear_map->constant().parameter_id()].push_back(
        linear_map->mutable_constant());
  }

  for (LinearMap& arg : *linear_map->mutable_arg())
    InitParameterMap(&arg);
}

Stat* Solver::GetStat(const std::string& name) {
  std::lock_guard<std::mutex> l(mutex_);

  auto iter = stat_map_.find(name);
  if (iter != stat_map_.end())
    return iter->second.get();

  Stat* stat = new StatImpl(name, &timer_);
  stat_map_.insert(make_pair(name, std::unique_ptr<Stat>(stat)));
  return stat;
}

std::vector<Stat*> Solver::GetStats(const std::string& prefix) {
  std::lock_guard<std::mutex> l(mutex_);

  std::vector<Stat*> retval;
  for (const auto& iter : stat_map_) {
    if (StartsWith(prefix, iter.first)) {
      retval.push_back(iter.second.get());
    }
  }

  return retval;
}

void Solver::UpdateStatus(const SolverStatus& status) {
  {
    std::lock_guard<std::mutex> l(mutex_);
    status_ = status;
  }
  if (status_callback_)
    status_callback_(status_);
}

SolverStatus Solver::status() {
  std::lock_guard<std::mutex> l(mutex_);
  return status_;
}

bool Solver::HasExternalStop() {
  if (!stop_callback_)
    return false;

  return stop_callback_();
}

void Solver::SetParameterValue(
    const std::string& parameter_id, const Constant& value) {
  LOG(INFO) << parameter_id << "\n" << value.DebugString();
  auto iter = parameter_map_.find(parameter_id);
  CHECK(iter != parameter_map_.end());
  for (Constant* constant : iter->second) {
    *constant = value;
  }
}
