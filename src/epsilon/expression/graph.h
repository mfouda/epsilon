#ifndef EXPRESSION_GRAPH_H
#define EXPRESSION_GRAPH_H

#include <vector>

#include "distopt/problem.pb.h"

class ProblemBuilder {
 public:
  // Problem builder with specified prefix
  ProblemBuilder(
      Problem* problem,
      const std::string& prefix) :
      problem_(problem),
      prefix_(prefix),
      variable_count_(0),
      constraint_count_(0) {}

  // Adds a variable in Expression
  Expression* AddVariable(const Size& size, Expression* expr);

  // Ads a new constraint
  Constraint* AddConstraint(const Cone& cone);

 private:
  Problem* problem_;  // Not owned
  std::string prefix_;

  int variable_count_;
  int constraint_count_;
};

void ComputeGraph(
    const Expression& input,
    Expression* output,
    ProblemBuilder* builder);

#endif  // EXPRESSION_GRAPH_H
