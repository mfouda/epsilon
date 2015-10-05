#include "epsilon/affine/split.h"

#include <glog/logging.h>

bool ShouldSplitNode(const Expression& expr) {
  return expr.expression_type() == Expression::ADD;
}

SplitExpressionIterator::SplitExpressionIterator(const Expression& expression)
    : done_(false),
      prev_(nullptr) {
  stack_.push_back({&expression, 0});

  // Special case for single node
  if (!ShouldSplitNode(expression)) {
    current_ = &expression;
    FillChain();
  } else {
    NextValue();
  }
}

void SplitExpressionIterator::NextValue() {
  CHECK(!done_);

  bool terminal = false;
  while (!terminal && !stack_.empty()) {
    std::pair<const Expression*, int>& item = stack_.back();
    current_ = item.first;

    if (ShouldSplitNode(*current_)) {
      // Split this node
      if (item.second == current_->arg_size()) {
        // Ascending, done with this node
        stack_.pop_back();
      } else if (prev_ == &current_->arg(item.second)) {
        // Ascending, Next visit sibling
        item.second++;
      } else {
        // Descending
        stack_.push_back({&current_->arg(item.second), 0});
      }
    } else {
      // No splitting, at a terminal node
      terminal = true;
      FillChain();
      stack_.pop_back();
    }

    prev_ = current_;
  }

  done_ = stack_.empty();
}

void SplitExpressionIterator::FillChain() {
  chain_.CopyFrom(*stack_[0].first);
  Expression* e = &chain_;
  for (int i = 1; i < stack_.size(); i++) {
    e->clear_arg();
    e->add_arg()->CopyFrom(*stack_[i].first);
    e = e->mutable_arg(0);
  }
}
