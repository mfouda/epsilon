
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "epsilon/file/file.h"

GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  int retval = RUN_ALL_TESTS();
  return retval;
}
