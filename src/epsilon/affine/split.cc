#include "epsilon/affine/split.h"

#include <glog/logging.h>

int ArgIndex(const Expression& expr) {
  if (expr.expression_type() == Expression::MULTIPLY ||
      expr.expression_type() == Expression::MULTIPLY_ELEMENTWISE) {
    // LHS is an expression which is a parameter to MULTIPLY operators, this
    // should not be split out.
    CHECK_EQ(2, expr.arg_size());
    return 1;
  }
  return 0;
}

SplitExpressionIterator::SplitExpressionIterator(const Expression& expression)
    : done_(false),
      prev_(nullptr) {
  stack_.push_back({&expression, ArgIndex(expression)});

  // Special case for single node w/ no splitting
  if (expression.arg_size() == 0) {
    current_ = &expression;
    FillChain();
  } else {
    NextValue();
  }
}

void SplitExpressionIterator::NextValue() {
  CHECK(!done_);

  while (!stack_.empty()) {
    bool leaf = false;
    std::pair<const Expression*, int>& item = stack_.back();
    current_ = item.first;

    if (item.second == current_->arg_size()) {
      // Visit node, check if leaf
      if (current_->arg_size() == 0) {
        leaf = true;
        FillChain();
      }
      stack_.pop_back();
    } else {
      if (prev_ == &current_->arg(item.second)) {
        // Ascending the tree
        item.second++;
      } else {
        // Descending the tree
        const Expression* next = &current_->arg(item.second);
        stack_.push_back({next, ArgIndex(*next)});
      }
    }

    prev_ = current_;
    if (leaf)
      break;
  }

  done_ = stack_.empty();
}

void SplitExpressionIterator::FillChain() {
  chain_.CopyFrom(*stack_[0].first);
  Expression* e = &chain_;
  for (int i = 1; i < stack_.size(); i++) {
    if (e->expression_type() == Expression::MULTIPLY ||
        e->expression_type() == Expression::MULTIPLY_ELEMENTWISE) {
      e = e->mutable_arg(ArgIndex(*e));
    } else {
      e->clear_arg();
      e->add_arg()->CopyFrom(*stack_[i].first);
      e = e->mutable_arg(0);
    }
  }
}
