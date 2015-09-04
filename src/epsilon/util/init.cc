#include "distopt/util/init.h"

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <glog/logging.h>
#include <grpc/grpc.h>
#include <gflags/gflags.h>

#include "distopt/file/file.h"

void HandleSignal(int s) {
  LOG(INFO) << "Caught interrupt, exiting";
  google::ShutdownGoogleLogging();
  grpc_shutdown();
  file::Cleanup();

  exit(1);
}

void SetupSignalHandlers() {
  struct sigaction handler;

  handler.sa_handler = HandleSignal;
  sigemptyset(&handler.sa_mask);
  handler.sa_flags = 0;

  sigaction(SIGINT, &handler, NULL);
}

void Init(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  grpc_init();
  file::Init();
  SetupSignalHandlers();
}
