#include "distopt/parameters/parameter_server_impl.h"

#include <glog/logging.h>

#include "distopt/util/time.h"

using google::protobuf::RepeatedField;

grpc::Status ParameterServerImpl::Fetch(
    grpc::ServerContext* context,
    const FetchRequest* request,
    FetchResponse* response) {
  VLOG(3) << "Fetch request: " << request->ShortDebugString();
  std::lock_guard<std::mutex> l(lock_);
  auto iter = parameters_.find(request->id());
  if (iter != parameters_.end() &&
      iter->second.timestamp_usec() > request->min_timestamp_usec()) {
    *response = iter->second;
  }

  VLOG(3) << "Fetch response: " << response->ShortDebugString();
  return grpc::Status::OK;
}

grpc::Status ParameterServerImpl::Update(
    grpc::ServerContext* context,
    const UpdateRequest* request,
    UpdateResponse* response) {
  VLOG(3) << "Update " << request->ShortDebugString();
  std::lock_guard<std::mutex> l(lock_);
  auto iter = parameters_.find(request->id());
  if (iter == parameters_.end()) {
    *parameters_[request->id()].mutable_value() = request->delta();
  } else {
    RepeatedField<double>* a = iter->second.mutable_value()->mutable_value();
    const RepeatedField<double>* b = &request->delta().value();

    if (a->size() != b->size()) {
      return grpc::Status(
          grpc::StatusCode::FAILED_PRECONDITION, "Wrong size");
    }
    for (int i = 0; i < a->size(); i++) {
      *a->Mutable(i) += b->Get(i);
    }
  }

  parameters_[request->id()].set_timestamp_usec(WallTime_Usec());


  return grpc::Status::OK;
}
