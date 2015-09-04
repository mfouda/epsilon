
#include "distopt/algorithms/solver.h"

#include "distopt/stats.pb.h"
#include "distopt/util/string.h"

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

  void Fill(StatSeries* series) {
    std::lock_guard<std::mutex> l(lock_);
    *series = series_;
  }

private:
  const Timer* timer_;  // Not owned

  std::mutex lock_;
  StatSeries series_;
};

Solver::Solver() : problem_id_(std::hash<uint64_t>()(WallTime_Usec())) {}

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

void Solver::UpdateStatus(const ProblemStatus& status) {
  {
    std::lock_guard<std::mutex> l(mutex_);
    status_ = status;
  }
  if (status_callback_)
    status_callback_(status_);
}

ProblemStatus Solver::status() {
  std::lock_guard<std::mutex> l(mutex_);
  return status_;
}

bool Solver::HasExternalStop() {
  if (!stop_callback_)
    return false;

  return stop_callback_();
}
