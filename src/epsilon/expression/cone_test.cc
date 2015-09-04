
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <glog/logging.h>

#include "distopt/util/vector_testutil.h"

using Eigen::VectorXd;
using Eigen::Infinity;

namespace cone {
void ProjectSecondOrder(
    const std::vector<const Eigen::VectorXd*>& input,
    const std::vector<Eigen::VectorXd*>& output);
void ProjectSecondOrderElementwise(
    const std::vector<const Eigen::VectorXd*>& input,
    const std::vector<Eigen::VectorXd*>& output);
void ProjectNonNegative(
    const std::vector<const Eigen::VectorXd*>& input,
    const std::vector<Eigen::VectorXd*>& output);
}  // namespace cone


class NonNegativeTest : public testing::Test {
 protected:
  void ProjectNonNegative(
      const VectorXd& x,
      const VectorXd& expected_x) {
    VectorXd x_copy(x);

    // NOTE(mwytock): Tests the case with in place storage
    std::vector<const VectorXd*> input({&x_copy});
    std::vector<VectorXd*> output({&x_copy});
    cone::ProjectNonNegative(input, output);

    EXPECT_TRUE(VectorEquals(expected_x, x_copy, 1e-3));
  }
};

TEST_F(NonNegativeTest, Basic) {
  VectorXd x(5), y(5);
  x << 1, -2, 3, 4, 5;
  y << 1,  0, 3, 4, 5;
  ProjectNonNegative(x, y);
}

class SecondOrderTest : public testing::Test {
 protected:
  void ProjectSecondOrder(
      double t, const VectorXd& x,
      double expected_t, const VectorXd& expected_x) {
    VectorXd t_vec(1);
    VectorXd x_copy(x);
    t_vec << t;

    // NOTE(mwytock): Tests the case with in place storage
    std::vector<const VectorXd*> input({&t_vec, &x_copy});
    std::vector<VectorXd*> output({&t_vec, &x_copy});
    cone::ProjectSecondOrder(input, output);

    EXPECT_EQ(1, t_vec.size());
    EXPECT_NEAR(expected_t, t_vec(0), 1e-3);
    EXPECT_TRUE(VectorEquals(expected_x, x_copy, 1e-3));
  }
};

TEST_F(SecondOrderTest, Basic) {
  VectorXd x(2), y(2);
  x << 2, 1;
  y << 1.447, 0.723;
  ProjectSecondOrder(1, x, 1.618, y);
}

TEST_F(SecondOrderTest, NoChange) {
  VectorXd x(2);
  x << 2, 1;
  ProjectSecondOrder(5, x, 5, x);
}

TEST_F(SecondOrderTest, Zero) {
  VectorXd x(2), y(2);
  x << 2, 1;
  y << 0, 0;
  ProjectSecondOrder(-5, x, 0, y);
}

class SecondOrderElementwiseTest : public testing::Test {
 protected:
  void ProjectSecondOrderElementwise(
      VectorXd* t,
      std::vector<VectorXd*>& x,
      const VectorXd& expected_t,
      std::vector<const VectorXd*>& expected_x) {
     // NOTE(mwytock): Tests the case with in place storage
    std::vector<const VectorXd*> input({t});
    std::vector<VectorXd*> output({t});
    for (VectorXd* xi : x) {
      input.push_back(xi);
      output.push_back(xi);
    }
    cone::ProjectSecondOrderElementwise(input, output);

    EXPECT_TRUE(VectorEquals(expected_t, *t, 1e-3));
    for (int i = 0; i < x.size(); i++) {
      EXPECT_TRUE(VectorEquals(*expected_x[i], *x[i], 1e-3));
    }
  }
};

TEST_F(SecondOrderElementwiseTest, Basic) {
  VectorXd x1(2), x2(2), tx(2);
  VectorXd y1(2), y2(2), ty(2);

  tx << 1, 5;
  x1 << 2, 3;
  x2 << 1, 2;

  ty << 1.618, 5;
  y1 << 1.447, 3;
  y2 << 0.7232, 2;

  std::vector<VectorXd*> x({&x1, &x2});
  std::vector<const VectorXd*> y({&y1, &y2});
  ProjectSecondOrderElementwise(&tx, x, ty, y);
}
