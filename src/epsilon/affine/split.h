#ifndef EPSILON_AFFINE_SPLIT_H
#define EPSILON_AFFINE_SPLIT_H

#include <vector>

#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression_util.h"

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

#endif  // EPSILON_AFFINE_SPLIT_H
