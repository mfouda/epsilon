
#include "distopt/expression/cone.h"

#include <unordered_map>

#include <glog/logging.h>
#include <Eigen/Dense>

using Eigen::VectorXd;
using std::unordered_map;

namespace cone {

double SOCAlpha(double t, double norm) {
  if (norm <= t) {
    return -1;
  } else if (norm <= -t) {
    return 0;
  } else {
    return 0.5*(t + norm);
  }
}

void ProjectSecondOrder(
    const std::vector<const VectorXd*>& input,
    const std::vector<VectorXd*>& output) {
  CHECK_GE(input.size(), 2);
  CHECK_EQ(input[0]->size(), 1);

  double t = (*input[0])(0);
  double norm_squared = 0.0;
  for (int i = 1; i < input.size(); i++) {
    norm_squared += input[i]->dot(*input[i]);
  }
  double norm = sqrt(norm_squared);
  double alpha = SOCAlpha(t, norm);

  (*output[0])[0] = alpha < 0 ? t : alpha;
  for (int i = 1; i < input.size(); i++) {
    double scale = alpha >= 0 ? alpha/norm : 1;
    *output[i] = *input[i]*scale;
  }
}

void ProjectSecondOrderElementwise(
    const std::vector<const VectorXd*>& input,
    const std::vector<VectorXd*>& output) {
  CHECK_GE(input.size(), 2);
  for (int i = 1; i < input.size(); i++) {
    CHECK_EQ(input[0]->size(), input[i]->size());
  }

  for (int j = 0; j < input[0]->size(); j++) {
    double t = (*input[0])[j];
    double norm_squared = 0.0;
    for (int i = 1; i < input.size(); i++) {
      norm_squared += (*input[i])[j]*(*input[i])[j];
    }
    double norm = sqrt(norm_squared);
    double alpha = SOCAlpha(t, norm);

    (*output[0])[j] = alpha < 0 ? t : alpha;
    for (int i = 1; i < input.size(); i++) {
      double scale = alpha >= 0 ? alpha/norm : 1;
      (*output[i])[j] = (*input[i])[j]*scale;
    }
  }
}

void ProjectNonNegative(
    const std::vector<const VectorXd*>& input,
    const std::vector<VectorXd*>& output) {
  for (int i = 0; i < input.size(); i++) {
    *output[i] = input[i]->array().max(0);
  }
}

}  // namespace cone

unordered_map<int, VectorFunction> kConeProjections = {
  {NON_NEGATIVE, &cone::ProjectNonNegative},
  {SECOND_ORDER, &cone::ProjectSecondOrder},
  {SECOND_ORDER_ELEMENTWISE, &cone::ProjectSecondOrderElementwise},
};

VectorFunction GetConeProjection(const Cone& cone) {
  auto iter = kConeProjections.find(cone);
  if (iter == kConeProjections.end()) {
    LOG(FATAL) << "No projection found for cone " << cone;
  }
  return iter->second;
}

Cone GetDualCone(const Cone& cone) {
  if (cone == NON_NEGATIVE ||
      cone == SECOND_ORDER ||
      cone == SEMI_DEFINITE) {
    return cone;
  }

  if (cone == L1)
    return L_INFINITY;
  if (cone == L_INFINITY)
    return L1;

  LOG(FATAL) << "Unknown dual for cone " << cone;
}
