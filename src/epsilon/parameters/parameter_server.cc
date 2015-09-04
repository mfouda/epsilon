
#include <mutex>
#include <unordered_map>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpc++/server_credentials.h>
#include <grpc++/server_builder.h>
#include <grpc++/server.h>
#include <grpc/grpc.h>

#include "distopt/file/file.h"
#include "distopt/parameters.grpc.pb.h"
#include "distopt/parameters/parameter_server_impl.h"
#include "distopt/util/init.h"
#include "distopt/util/string.h"

DEFINE_int32(port, 8002, "Port to listen on");

int main(int argc, char** argv) {
  Init(argc, argv);

  ParameterServerImpl parameter_server;

  grpc::ServerBuilder builder;
  builder.AddListeningPort(
    StringPrintf("0.0.0.0:%d", FLAGS_port), grpc::InsecureServerCredentials());
  builder.RegisterService(&parameter_server);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Listening on port " << FLAGS_port;
  server->Wait();
}
