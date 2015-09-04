
#include "distopt/expression/linear.h"

#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <glog/logging.h>

#include "distopt/expression/cone.h"
#include "distopt/expression/expression.h"
#include "distopt/expression/graph.h"
#include "distopt/problem.pb.h"
#include "distopt/util/string.h"
#include "distopt/util/vector.h"

using Eigen::Map;
using google::protobuf::RepeatedPtrField;

void PromoteConstants(const Expression& parent, Expression* child) {
  if (child->expression_type() == Expression::CONSTANT &&
      child->constant().scalar() != 0) {
    child->mutable_size()->CopyFrom(parent.size());
  }
}

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

bool IsLinearOperator(const Expression& input) {
  return (input.expression_type() == Expression::ADD ||
          input.expression_type() == Expression::INDEX ||
	  input.expression_type() == Expression::MULTIPLY ||
	  input.expression_type() == Expression::NEGATE ||
          input.expression_type() == Expression::MULTIPLY_ELEMENTWISE ||
          input.expression_type() == Expression::SUM ||
          input.expression_type() == Expression::TRANSPOSE);
}


void CanonicalizeExpression(
  const Expression& input,
  Expression* output,
  ProblemBuilder* builder) {
  VLOG(2) << "CanonicalizeExpression\n" << input.DebugString();

  output->mutable_size()->CopyFrom(input.size());
  if (input.expression_type() == Expression::CONSTANT) {
    output->set_expression_type(Expression::CONSTANT);
    output->mutable_constant()->CopyFrom(input.constant());
  } else if (input.expression_type() == Expression::VARIABLE) {
    output->set_expression_type(Expression::VARIABLE);
    output->mutable_variable()->CopyFrom(input.variable());
  } else if (IsLinearOperator(input)) {
    // TODO(mwytock): Just have ComputeGraph() do all the work instead of these
    // two clauses
    output->set_expression_type(input.expression_type());
    for (const Expression& arg : input.arg()) {
      Expression* output_arg = output->add_arg();
      CanonicalizeExpression(arg, output_arg, builder);
      PromoteConstants(input, output_arg);
    }
    // Extra elements to copy over
    if (input.expression_type() == Expression::INDEX) {
      output->mutable_key()->CopyFrom(input.key());
    }
  } else {
    // Non-linear atom
    Expression canon_input;
    canon_input.CopyFrom(input);
    canon_input.clear_arg();
    for (int i = 0; i < input.arg_size(); i++) {
      Expression* canon_arg = canon_input.add_arg();
      CanonicalizeExpression(input.arg(i), canon_arg, builder);
      PromoteConstants(input, canon_arg);
    }
    ComputeGraph(canon_input, output, builder);
  }
}

void BuildConstraint(
    const Constraint& constraint,
    WorkerPool* worker_pool,
    LinearHashOp* A,
    HashVector* b,
    FunctionHashOp* K_star_proj) {
  VLOG(2) << "BuildConstraint";

  std::vector<std::string> output_keys;

  for (const Expression& expression : constraint.arg()) {
    std::string output_key =
        constraint.constraint_id() + ":" + std::to_string(output_keys.size());
    output_keys.push_back(output_key);

    SplitExpressionIterator iter(expression);
    for (; !iter.done(); iter.NextValue()) {
      if (iter.leaf().expression_type() == Expression::CONSTANT) {
        b->Insert(output_key,
                  EvaluateConstant(worker_pool, iter.chain(), iter.leaf()));
      } else {
        CHECK_EQ(Expression::VARIABLE, iter.leaf().expression_type());

        // Negate due to s = b - Ax
        Expression neg;
        neg.set_expression_type(Expression::NEGATE);
        *neg.mutable_size() = iter.chain().size();
        *neg.add_arg() = iter.chain();
        A->Insert(output_key, iter.leaf().variable().variable_id(), neg);
      }
    }
  }

  if (constraint.cone() != ZERO)
    K_star_proj->Insert(output_keys, GetConeProjection(constraint.cone()));
}

void BuildObjective(
    const Expression& expression,
    WorkerPool* worker_pool,
    HashVector* c,
    double* c0) {
  VLOG(2) << "BuildObjective";

  CHECK_EQ(2, expression.size().dim_size());
  CHECK_EQ(1, expression.size().dim(0));
  CHECK_EQ(1, expression.size().dim(1));

  SplitExpressionIterator iter(expression);
  for (; !iter.done(); iter.NextValue()) {
    if (iter.leaf().expression_type() == Expression::CONSTANT) {
      *c0 += EvaluateConstant(worker_pool, iter.chain(), iter.leaf())(0);
    } else {
      CHECK_EQ(Expression::VARIABLE, iter.leaf().expression_type());
      c->Insert(iter.leaf().variable().variable_id(),
                GetCoefficients(worker_pool, iter.chain()).transpose());
    }
  }
}

void BuildConeProblem(
    WorkerPool* worker_pool,
    const Problem& input,
    LinearHashOp* A,
    HashVector* b,
    HashVector* c,
    double* c0,
    FunctionHashOp* K_star_proj) {
  Problem canon;
  CanonicalizeLinearConeProblem(input, &canon);

  // Build hash version
  BuildObjective(canon.objective(), worker_pool, c, c0);
  for (const Constraint& constraint : canon.constraint()) {
    BuildConstraint(constraint, worker_pool, A, b, K_star_proj);
  }
}

void CanonicalizeLinearConeProblem(const Problem& input, Problem* output) {
  VLOG(2) << "Input problem\n" << input.DebugString();

  ProblemBuilder builder(output, "linear_cone");
  CanonicalizeExpression(
      input.objective(), output->mutable_objective(), &builder);
  for (const Constraint& constraint : input.constraint()) {
    Constraint* canon_constraint = output->add_constraint();
    canon_constraint->set_cone(constraint.cone());
    canon_constraint->set_constraint_id(constraint.constraint_id());
    for (const Expression& expr : constraint.arg()) {
      CanonicalizeExpression(expr, canon_constraint->add_arg(), &builder);
    }
  }
  VLOG(2) << "Linear cone problem\n" << output->DebugString();
}
