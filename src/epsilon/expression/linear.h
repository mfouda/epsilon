#ifndef EXPRESSION_LINEAR_H
#define EXPRESSION_LINEAR_H

class WorkerPool;

#include "distopt/hash/hash.h"
#include "distopt/problem.pb.h"

class SplitExpressionIterator {
 public:
  SplitExpressionIterator(const Expression& expression);

  // Accessors
  bool done() { return done_; }
  const Expression& chain() { return chain_; }
  const Expression& leaf() { return *current_; }

  // Move to next value
  void NextValue();

 private:
  void FillChain();

  bool done_;
  std::vector<std::pair<const Expression*, int> > stack_;
  Expression chain_;
  const Expression* current_;
  const Expression* prev_;
};

void CanonicalizeLinearConeProblem(
    const Problem& input,
    Problem* output);

#endif  // EXPRESSION_LINEAR_H
