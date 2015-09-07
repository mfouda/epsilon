#if USE_GLOG
#include <glog/logging.h>
#else

#ifndef EPSILON_UTIL_GLOG_H
#define EPSILON_UTIL_GLOG_H

#include <iosfwd>

struct nullstream {};

// Swallow all types
template <typename T>
nullstream& operator<<(nullstream & s, T const &) {return s;}

static nullstream logstream;
#define CHECK(x) logstream
#define CHECK_EQ(x, y) logstream
#define CHECK_GE(x, y) logstream
#define CHECK_GT(x, y) logstream
#define CHECK_LE(x, y) logstream

#define VLOG(n) logstream
#define LOG(n) logstream

#endif  // EPSILON_UTIL_GLOG_H

#endif  // USE_GLOG
