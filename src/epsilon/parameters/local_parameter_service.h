#ifndef DISTOPT_PARAMETERS_LOCAL_PARAMETER_SERVICE_H
#define DISTOPT_PARAMETERS_LOCAL_PARAMETER_SERVICE_H

#include "distopt/parameters/parameter_service.h"

class LocalParameterService final : public ParameterService {
public:
  LocalParameterService();
  ~LocalParameterService();

  void UpdateAsync(uint64_t parameter_id, Eigen::VectorXd delta) override;
  void FetchAsync(uint64_t parameter_id) override;
  void SetFetchCallback(
      std::function<void(uint64_t, Eigen::VectorXd value)> callback) override {
    fetch_callback_ = callback;
  }
  void HandleAsyncCompletions() override;

  Eigen::VectorXd Update(uint64_t parameter_id, Eigen::VectorXd delta) override;
  Eigen::VectorXd Fetch(uint64_t parameter_id) override;

private:
    std::function<void(uint64_t, Eigen::VectorXd value)> fetch_callback_;

};

#endif  // DISTOPT_PARAMETERS_LOCAL_PARAMETER_SERVICE_H
