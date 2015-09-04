#ifndef DISTOPT_UTIL_BACKENDS_TESTUTIL_H
#define DISTOPT_UTIL_BACKENDS_TESTUTIL_H

#include "distopt/parameters.grpc.pb.h"
#include "distopt/worker/worker_pool.h"

// TODO(mwytock): These backends are just used for testing currently but we want
// to make them a first-class citizen and use local unix sockest (supported by
// grpc) in order to run solvers in a completely local mode
void InitLocalBackends();
void DestroyLocalBackends();

class LocalWorkerPool final : public WorkerPool {
 public:
  std::shared_ptr<Worker::StubInterface> GetWorkerStub(
      const std::string& key);
  std::shared_ptr<SolverService::StubInterface> GetSolverStub(
      const std::string& key);
};

std::string GetLocalServerAddress();

#endif  // DISTOPT_UTIL_BACKENDS_TESTUTIL_H
