
#include <sys/time.h>
#include <stdint.h>

#include <glog/logging.h>


uint64_t WallTime_Usec() {
  struct timeval time;
  CHECK_EQ(0, gettimeofday(&time, nullptr));
  return time.tv_sec*1000000 + time.tv_usec;
}

double WallTime() {
  return WallTime_Usec()*1e-6;
}
