#include "distopt/expression/eval.h"

#include <unordered_map>
#include <memory>

#include <glog/logging.h>

#include "distopt/expression/expression.h"
#include "distopt/expression/problem.h"
#include "distopt/util/vector.h"

namespace eval {

Eigen::MatrixXd LogDet(
    const Expression& expr,
    const std::vector<Eigen::MatrixXd>& args) {
  CHECK_EQ(1, args.size());
  Eigen::VectorXcd v = args[0].eigenvalues();
  if ((v.imag().array() != 0).any() ||
      (v.real().array() <= 0).any()) {
    return MatrixXd::Constant(1, 1, -INFINITY);
  }
  return MatrixXd::Constant(1, 1, v.real().array().log().sum());
}

Eigen::MatrixXd Negate(
    const Expression& expr,
    const std::vector<Eigen::MatrixXd>& args) {
  CHECK_EQ(1, args.size());
  return -args[0];
}

Eigen::MatrixXd Sum(
    const Expression& expr,
    const std::vector<Eigen::MatrixXd>& args) {
  CHECK_EQ(1, args.size());
  return Eigen::MatrixXd::Constant(1, 1, args[0].sum());
}

Eigen::MatrixXd PNorm(
    const Expression& expr,
    const std::vector<Eigen::MatrixXd>& args) {
  CHECK_EQ(1, args.size());
  const double p = expr.p();
  CHECK_GE(p, 1);
  return Eigen::MatrixXd::Constant(
      1, 1, pow(args[0].array().abs().pow(p).sum(), 1/p));
}

Eigen::MatrixXd Power(
    const Expression& expr,
    const std::vector<Eigen::MatrixXd>& args) {
  CHECK_EQ(1, args.size());
  return args[0].array().pow(expr.p());
}

Eigen::MatrixXd Index(
    const Expression& expr,
    const std::vector<Eigen::MatrixXd>& args) {
  CHECK_EQ(1, args.size());
  CHECK_EQ(2, expr.key_size());

  // TODO(mwytock): Support general step sizes
  CHECK_EQ(1, expr.key(0).step());
  CHECK_EQ(1, expr.key(1).step());

  return args[0].block(
      expr.key(0).start(),
      expr.key(1).start(),
      expr.key(0).stop() - expr.key(0).start(),
      expr.key(1).stop() - expr.key(1).start());
}

Eigen::MatrixXd Multiply(
    const Expression& expr,
    const std::vector<Eigen::MatrixXd>& args) {
  CHECK_GT(args.size(), 0);
  Eigen::MatrixXd A = args[0];
  for (int i = 1; i < args.size(); i++)
    A *= args[i];
  return A;
}

Eigen::MatrixXd MultiplyElementwise(
    const Expression& expr,
    const std::vector<Eigen::MatrixXd>& args) {
  CHECK_GT(args.size(), 0);
  Eigen::MatrixXd A = args[0];
  for (int i = 1; i < args.size(); i++)
    A = A.array() * args[i].array();
  return A;;
}

Eigen::MatrixXd Add(
    const Expression& expr,
    const std::vector<Eigen::MatrixXd>& args) {
  CHECK_GT(args.size(), 0);
  Eigen::MatrixXd A = args[0];
  for (int i = 1; i < args.size(); i++)
    A += args[i];
  return A;
}

// L2 norm applied to each arg
Eigen::MatrixXd Norm2Elementwise(
    const Expression& expr,
    const std::vector<Eigen::MatrixXd>& args) {
  CHECK_GT(args.size(), 0);

  Eigen::MatrixXd retval = Eigen::MatrixXd::Zero(
      args[0].rows(), args[0].cols());
  for (const Eigen::MatrixXd& arg : args) {
    CHECK_EQ(arg.rows(), retval.rows());
    CHECK_EQ(arg.cols(), retval.cols());
    retval.array() += arg.array().square();
  }
  return retval.array().sqrt();
}

}  // namespace eval

typedef Eigen::MatrixXd(*EvalFunction)(
    const Expression& expr,
    const std::vector<Eigen::MatrixXd>& args);

std::unordered_map<int, EvalFunction> kEvalFunctions = {
  // 1 arg
  {Expression::INDEX, &eval::Index},
  {Expression::LOG_DET, &eval::LogDet},
  {Expression::NEGATE, &eval::Negate},
  {Expression::POWER, &eval::Power},
  {Expression::P_NORM, &eval::PNorm},
  {Expression::SUM, &eval::Sum},

  // Multiple args
  {Expression::ADD, &eval::Add},
  {Expression::MULTIPLY, &eval::Multiply},
  {Expression::MULTIPLY_ELEMENTWISE, &eval::MultiplyElementwise},
  {Expression::NORM_2_ELEMENTWISE, &eval::Norm2Elementwise},
};

Eigen::MatrixXd ExpressionEvaluator::Evaluate(
    uint64_t problem_id,
    const Expression& expr) {
  VLOG(2) << "Evaluate\n" << expr.DebugString();

  if (expr.expression_type() == Expression::VARIABLE) {
    // Base case for variable
    return ToMatrix(
        parameter_service_->Fetch(
            VariableId(problem_id, expr.variable().variable_id())),
        GetDimension(expr, 0),
        GetDimension(expr, 1));
  } else if (expr.expression_type() == Expression::CONSTANT) {
    // Base case for constant
    if (expr.constant().data_location() != "") {
      std::unique_ptr<const Data> d = ReadSplitData(
          expr.constant().data_location());
      return GetMatrixData(*d);
    } else {
      return Eigen::MatrixXd::Constant(
          GetDimension(expr, 0),
          GetDimension(expr, 1),
          expr.constant().scalar());
    }
  } else {
    // Recursive case
    auto iter = kEvalFunctions.find(expr.expression_type());
    if (iter == kEvalFunctions.end()) {
      LOG(FATAL) << "No eval function for "
                 << Expression::Type_Name(expr.expression_type());
    }

    std::vector<Eigen::MatrixXd> args(expr.arg_size());
    for (int i = 0; i < expr.arg_size(); i++) {
      args[i] = Evaluate(problem_id, expr.arg(i));
    }
    return iter->second(expr, args);
  }
}
