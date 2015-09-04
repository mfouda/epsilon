#ifndef UTIL_SYNCHRONIZATION_H
#define UTIL_SYNCHRONIZATION_H

#include <condition_variable>
#include <mutex>

#include <google/protobuf/stubs/common.h>

using google::protobuf::Closure;
using google::protobuf::NewCallback;

class BlockingCounter {
 public:
  BlockingCounter() : count_(0) {}
  explicit BlockingCounter(int count) : count_(count) {}

  void Decrement();
  void Increment();
  void Wait();

 private:
  int count_;
  std::mutex lock_;
  std::condition_variable zero_;
};

class BlockingClosure : public Closure {
public:
  BlockingClosure() : done_(false) {}
  virtual void Run();
  void Wait();

 private:
  bool done_;
  std::mutex lock_;
  std::condition_variable done_cv_;
};


class AutoClosureRunner {
 public:
  explicit AutoClosureRunner(Closure* closure)
      : closure_(closure) {}
  ~AutoClosureRunner() { closure_->Run(); }

 private:
  Closure* closure_;
};


Closure* NewLambdaCallback(std::function<void()> f);


#endif  // UTIL_SYNCHRONIZATION_H
