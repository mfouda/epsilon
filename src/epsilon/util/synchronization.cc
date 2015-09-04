#include <glog/logging.h>

#include "distopt/util/synchronization.h"

void BlockingCounter::Decrement() {
  std::unique_lock<std::mutex> l(lock_);
  CHECK_GT(count_, 0);
  if (--count_ == 0)
    zero_.notify_all();
}

void BlockingCounter::Increment() {
  std::unique_lock<std::mutex> l(lock_);
  ++count_;
}

void BlockingCounter::Wait() {
  std::unique_lock<std::mutex> l(lock_);
  if (count_ == 0)
    return;
  zero_.wait(l);
}

void BlockingClosure::Run() {
  std::unique_lock<std::mutex> l(lock_);
  done_ = true;
  done_cv_.notify_all();
}

void BlockingClosure::Wait() {
  std::unique_lock<std::mutex> l(lock_);
  if (done_)
    return;
  done_cv_.wait(l);
}

class LambdaClosure : public Closure {
 public:
  LambdaClosure(std::function<void()> f) : f_(f) {}
  virtual void Run() { f_(); delete this; }

 private:
  std::function<void()> f_;
};

Closure* NewLambdaCallback(std::function<void()> f) {
  return new LambdaClosure(f);
}
