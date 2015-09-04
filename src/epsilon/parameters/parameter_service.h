#ifndef EPSILON_PARAMETERS_PARAMETER_SERVICE_H
#define EPSILON_PARAMETERS_PARAMETER_SERVICE_H

#include <Eigen/Dense>

// Interface for parameter service, either remote or local (coming soon).
class ParameterService {
 public:
  virtual ~ParameterService() {}

  // Async interface
  virtual void UpdateAsync(uint64_t parameter_id, Eigen::VectorXd delta) = 0;
  virtual void FetchAsync(uint64_t parameter_id) = 0;
  virtual void SetFetchCallback(
      std::function<void(uint64_t, Eigen::VectorXd value)> done) = 0;
  virtual void HandleAsyncCompletions() = 0;

  // Sync interface
  virtual Eigen::VectorXd Update(uint64_t parameter_id, Eigen::VectorXd delta) = 0;
  virtual Eigen::VectorXd Fetch(uint64_t parameter_id) = 0;

};

#endif  // EPSILON_PARAMETERS_PARAMETER_SERVICE_H
