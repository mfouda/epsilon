#ifndef ALGORITHMS_SOLVER_H
#define ALGORITHMS_SOLVER_H

#include <stdint.h>
#include <time.h>

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <glog/logging.h>

#include "epsilon/util/time.h"
#include "epsilon/solver.pb.h"

class Solution;
class WorkerPool;

class Stat {
 public:
  virtual void AddValue(double value) = 0;
  virtual void Fill(SolverStatSeries* series) = 0;
};

class Timer {
 public:
  Timer() : start_(WallTime_Usec()) {}

  // TODO(mwytock): Support pause here

  uint64_t GetTimeUsec() const {
    return WallTime_Usec() - start_;
  }

 private:
  uint64_t start_;
};

class Solver {
 public:
  Solver();
  virtual ~Solver() {}

  // Implemented by sub classes
  virtual void Solve() = 0;

  // Returns current problem status
  SolverStatus status();

  // Allows callers to get notifications when status changes
  void RegisterStatusCallback(
      std::function<void(const SolverStatus&)> callback) {
    status_callback_ = callback;
  }

  // Allows solver to stop according to an external signal.
  void RegisterStopCallback(
      std::function<bool()> callback) {
    stop_callback_ = callback;
  }

  // Get a stat for both read/write
  // TODO(mwytock): We should separate out read/write interfaces here and make
  // the write interface protected ala UpdateStatus()
  Stat* GetStat(const std::string& id);
  std::vector<Stat*> GetStats(const std::string& prefix);

  uint64_t problem_id() { return problem_id_; }
  void set_problem_id(uint64_t problem_id) { problem_id_ = problem_id; }

 protected:
  // Update solution status
  void UpdateStatus(const SolverStatus& status);

  // True if an external stop was requested
  bool HasExternalStop();

 private:
  std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<Stat> > stat_map_;
  SolverStatus status_;
  Timer timer_;

  uint64_t problem_id_;

  std::function<bool()> stop_callback_;
  std::function<void(const SolverStatus&)> status_callback_;
};

#endif  // ALGORITHMS_SOLVER_H
