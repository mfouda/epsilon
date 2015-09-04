#include "distopt/util/backends_testutil.h"

#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpc++/channel_arguments.h>
#include <grpc++/create_channel.h>
#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <grpc++/server_credentials.h>

#include "distopt/algorithms/solver_service_impl.h"
#include "distopt/parameters/parameter_server_impl.h"
#include "distopt/util/port.h"
#include "distopt/util/string.h"
#include "distopt/util/thread_pool.h"
#include "distopt/worker/worker_impl.h"

DEFINE_int32(local_server_threads, 10, "LocalWorkerPool threads");

// A local server that listens on a random port and exports all the necessary
// backend interfaces we need.
static std::string local_addr;
static std::unique_ptr<grpc::Server> local_server;
static std::unique_ptr<ThreadPool> thread_pool;

// Service interfaces
static std::unique_ptr<Worker::Service> worker_service;
static std::unique_ptr<SolverService::Service> worker_solver_service;
static std::unique_ptr<ParameterServer::Service> parameter_server;

// Stub/client interfaces
static std::shared_ptr<Worker::StubInterface> worker_stub;
static std::shared_ptr<SolverService::StubInterface> worker_solver_stub;
static std::shared_ptr<ParameterServer::StubInterface> parameter_server_stub;

void InitLocalBackends() {
  local_addr = StringPrintf("localhost:%d", PickUnusedPortOrDie());

  // Create the stubs
  std::shared_ptr<grpc::ChannelInterface> channel(
      grpc::CreateChannel(
          local_addr, grpc::InsecureCredentials(), grpc::ChannelArguments()));
  parameter_server_stub = std::move(ParameterServer::NewStub(channel));
  worker_stub = std::move(Worker::NewStub(channel));
  worker_solver_stub = std::move(SolverService::NewStub(channel));

  // Create the services
  parameter_server.reset(new ParameterServerImpl());
  worker_service.reset(new WorkerImpl());
  worker_solver_service.reset(new SolverServiceImpl(
      std::bind(CreateSolver_Worker, parameter_server_stub.get(),
                std::placeholders::_1)));

  // Create the server
  thread_pool.reset(new ThreadPool(FLAGS_local_server_threads));
  grpc::ServerBuilder builder;
  builder.AddListeningPort(local_addr, grpc::InsecureServerCredentials());
  builder.RegisterService(parameter_server.get());
  builder.RegisterService(worker_service.get());
  builder.RegisterService(worker_solver_service.get());
  builder.SetThreadPool(thread_pool.get());
  local_server = builder.BuildAndStart();
}

void DestroyLocalBackends() {
  local_server.reset();

  worker_solver_service.reset();
  worker_service.reset();
  parameter_server.reset();

  worker_solver_stub.reset();
  worker_stub.reset();
  parameter_server_stub.reset();
}

std::shared_ptr<Worker::StubInterface> LocalWorkerPool::GetWorkerStub(
    const std::string& key) {
  return worker_stub;
}

std::shared_ptr<SolverService::StubInterface> LocalWorkerPool::GetSolverStub(
    const std::string& key) {
  return worker_solver_stub;
}

std::string GetLocalServerAddress() {
  return local_addr;
}
