#include "epsilon/parameters/local_parameter_service.h"

#include <memory>
#include <mutex>
#include <unordered_map>

#include "epsilon/util/logging.h"

class LocalParameters {
public:
  void Update(uint64_t parameter_id, Eigen::VectorXd delta) {
    std::lock_guard<std::mutex> l(lock_);
    auto iter = params_.find(parameter_id);
    if (iter == params_.end()) {
      params_[parameter_id] = delta;
    } else {
      CHECK_EQ(iter->second.size(), delta.size());
      iter->second += delta;
    }
  }

  Eigen::VectorXd Fetch(uint64_t parameter_id) {
    std::lock_guard<std::mutex> l(lock_);
    auto iter = params_.find(parameter_id);
    if (iter == params_.end())
      return Eigen::VectorXd::Constant(1, 0);
    return iter->second;
  }

private:
  std::mutex lock_;
  std::unordered_map<uint64_t, Eigen::VectorXd> params_;
};

LocalParameters parameters;

LocalParameterService::LocalParameterService() {}

LocalParameterService::~LocalParameterService() {}

void LocalParameterService::UpdateAsync(
    uint64_t parameter_id, Eigen::VectorXd delta) {
  Update(parameter_id, delta);
}

void LocalParameterService::FetchAsync(
    uint64_t parameter_id) {
  if (!fetch_callback_)
    return;
  fetch_callback_(parameter_id, parameters.Fetch(parameter_id));
}

void LocalParameterService::HandleAsyncCompletions() {}

Eigen::VectorXd LocalParameterService::Update(
    uint64_t parameter_id, Eigen::VectorXd delta) {
  parameters.Update(parameter_id, delta);
  return parameters.Fetch(parameter_id);
}

Eigen::VectorXd LocalParameterService::Fetch(uint64_t parameter_id) {
  return parameters.Fetch(parameter_id);
}
