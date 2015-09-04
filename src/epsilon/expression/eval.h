#ifndef DISTOPT_EXPRESSION_EVAL
#define DISTOPT_EXPRESSION_EVAL

#include <memory>

#include <Eigen/Dense>

#include "distopt/expression.pb.h"
#include "distopt/parameters/parameter_service.h"

class ExpressionEvaluator {
 public:
  ExpressionEvaluator(
      std::unique_ptr<ParameterService> parameter_service)
      : parameter_service_(std::move(parameter_service)) {}

  Eigen::MatrixXd Evaluate(uint64_t problem_id, const Expression& expression);

 private:
  std::unique_ptr<ParameterService> parameter_service_;
};

#endif  // DISTOPT_EXPRESSION_EVAL
