
#include "distopt/expression/graph.h"

#include <unordered_map>

#include <glog/logging.h>

#include "distopt/expression/expression.h"

// Nasty temp hack just to experiment w/ l1 cone
DEFINE_bool(use_l1_cone, false, "Use L1 cone");

namespace graph {

// Graph implementation of non-linear operators
void Abs(
    const Expression& input, Expression* output, ProblemBuilder* builder) {
  CHECK_EQ(1, input.arg_size());
  const Expression& x = input.arg(0);

  builder->AddVariable(x.size(), output);

  // t >= -x
  Expression* expr = builder->AddConstraint(Cone::NON_NEGATIVE)->add_arg();
  expr->set_expression_type(Expression::ADD);
  expr->mutable_size()->CopyFrom(x.size());
  *expr->add_arg() = *output;
  *expr->add_arg() = x;

  // t >= x
  expr = builder->AddConstraint(Cone::NON_NEGATIVE)->add_arg();
  expr->set_expression_type(Expression::ADD);
  expr->mutable_size()->CopyFrom(x.size());
  *expr->add_arg() = *output;
  *expr->add_arg() = expression::Negate(x);
}

void Power(
    const Expression& input, Expression* output, ProblemBuilder* builder) {
  CHECK_EQ(1, input.arg_size());
  CHECK_EQ(2, input.p());
  const Expression& x = input.arg(0);

  // t >= x^2 is equivalent to constraint norm2([1 - t, 2x]) <= 1 + t
  builder->AddVariable(x.size(), output);
  Constraint* constraint = builder->AddConstraint(Cone::SECOND_ORDER);

  {
    // 1 + t
    Expression* expr = constraint->add_arg();
    expr->set_expression_type(Expression::ADD);
    expr->mutable_size()->CopyFrom(x.size());
    *expr->add_arg() = expression::ScalarConstant(1);
    *expr->add_arg() = *output;
  }

  {
    // 1 - t
    Expression* expr = constraint->add_arg();
    expr->set_expression_type(Expression::ADD);
    expr->mutable_size()->CopyFrom(x.size());
    *expr->add_arg() = expression::ScalarConstant(1);
    *expr->add_arg() = expression::Negate(*output);
  }

  {
    // 2x
    Expression* expr = constraint->add_arg();
    expr->set_expression_type(Expression::MULTIPLY);
    expr->mutable_size()->CopyFrom(x.size());
    *expr->add_arg() = expression::ScalarConstant(2);
    *expr->add_arg() = x;
  }
}

void Norm1(
    const Expression& input, Expression* output, ProblemBuilder* builder) {
  if (FLAGS_use_l1_cone) {
    builder->AddVariable(input.size(), output);

    Constraint* constraint = builder->AddConstraint(Cone::L1);
    *constraint->add_arg() = *output;
    for (const Expression& arg : input.arg()) {
      *constraint->add_arg() = arg;
    }
  } else {
    Expression abs_x;
    Abs(input, &abs_x, builder);
    *output = expression::Sum(abs_x);
  }
}

void Norm2(
    const Expression& input, Expression* output, ProblemBuilder* builder) {
  builder->AddVariable(input.size(), output);

  Constraint* constraint = builder->AddConstraint(Cone::SECOND_ORDER);
  *constraint->add_arg() = *output;
  for (const Expression& arg : input.arg()) {
    *constraint->add_arg() = arg;
  }
}

void PNorm(
    const Expression& input, Expression* output, ProblemBuilder* builder) {
  if (input.p() == 1) {
    Norm1(input, output, builder);
  } else {
    CHECK_EQ(input.p(), 2);
    Norm2(input, output, builder);
  }
}

void Norm2Elementwise(
    const Expression& input, Expression* output, ProblemBuilder* builder) {
  builder->AddVariable(input.size(), output);

  Constraint* constraint = builder->AddConstraint(Cone::SECOND_ORDER_ELEMENTWISE);
  *constraint->add_arg() = *output;
  for (const Expression& arg : input.arg()) {
    *constraint->add_arg() = arg;
  }
}

}  // namespace graph



typedef void(*GraphFunction)(
    const Expression& input,
    Expression* output,
    ProblemBuilder* builder);
std::unordered_map<int, GraphFunction> kGraphFunctions = {
  {Expression::ABS, &graph::Abs},
  {Expression::NORM_2_ELEMENTWISE, &graph::Norm2Elementwise},
  {Expression::POWER, &graph::Power},
  {Expression::P_NORM, &graph::PNorm},
};

void ComputeGraph(
    const Expression& input,
    Expression* output,
    ProblemBuilder* builder) {
  VLOG(2) << "ComputeGraph " << input.expression_type();

  auto iter = kGraphFunctions.find(input.expression_type());
  if (iter == kGraphFunctions.end()) {
    LOG(FATAL) << "No graph implementation for " << input.expression_type();
  }

  iter->second(input, output, builder);
}

Expression* ProblemBuilder::AddVariable(const Size& size, Expression* x) {
  const std::string variable_id =
      prefix_ + ":var:" + std::to_string(variable_count_++);
  VLOG(2) << "AddVariable " << variable_id
          << " (" << size.ShortDebugString() << ")";

  x->set_expression_type(Expression::VARIABLE);
  x->mutable_size()->CopyFrom(size);
  x->mutable_variable()->set_variable_id(variable_id);
  return x;
}

Constraint* ProblemBuilder::AddConstraint(const Cone& cone) {
  const std::string constraint_id =
      prefix_ + ":constraint:" + std::to_string(constraint_count_++);
  VLOG(2) << "AddConstraint " << constraint_id
          << " (" << Cone_Name(cone) << ")";

  Constraint* constraint = problem_->add_constraint();
  constraint->set_cone(cone);
  constraint->set_constraint_id(constraint_id);
  return constraint;
}
