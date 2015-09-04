#ifndef EXPRESSION_CONE_H
#define EXPRESSION_CONE_H

#include <Eigen/Dense>

#include "distopt/problem.pb.h"
#include "distopt/util/vector.h"

VectorFunction GetConeProjection(const Cone& cone);

Cone GetDualCone(const Cone& cone);

#endif  // EXPRESSION_CONE_H
