#ifndef PARAMETERS_PARAMETER_SERVER_IMPL
#define PARAMETERS_PARAMETER_SERVER_IMPL

#include <mutex>
#include <unordered_map>

#include "distopt/parameters.grpc.pb.h"

class ParameterServerImpl final : public ParameterServer::Service {
private:
  grpc::Status Fetch(
      grpc::ServerContext* context,
      const FetchRequest* request,
      FetchResponse* response);

  grpc::Status Update(
      grpc::ServerContext* context,
      const UpdateRequest* request,
      UpdateResponse* response);

  std::mutex lock_;
  std::unordered_map<uint64_t, FetchResponse> parameters_;
};

#endif  // PARAMETERS_PARAMETER_SERVER_IMPL
