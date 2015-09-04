
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpc/grpc.h>
#include <gtest/gtest.h>

#include "epsilon/file/file.h"

GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  grpc_init();
  file::Init();

  int retval = RUN_ALL_TESTS();

  grpc_shutdown();
  file::Cleanup();
  return retval;
}
