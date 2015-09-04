
#include <stdlib.h>

#include <memory>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "distopt/expression/expression.h"
#include "distopt/expression/expression_testutil.h"
#include "distopt/expression/linear.h"
#include "distopt/hash/hash.h"
#include "distopt/problem.pb.h"
#include "distopt/util/backends_testutil.h"
#include "distopt/util/problems.h"
#include "distopt/util/vector_testutil.h"

using Eigen::Map;

TEST(SplitExpressionIterator, Single) {
  Expression e;
  e.set_expression_type(Expression::CONSTANT);
  SplitExpressionIterator iter(e);

  ASSERT_FALSE(iter.done());
  EXPECT_EQ(Expression::CONSTANT, iter.leaf().expression_type());
  EXPECT_EQ(0, iter.leaf().arg_size());
  EXPECT_EQ(Expression::CONSTANT, iter.chain().expression_type());
  EXPECT_EQ(0, iter.chain().arg_size());

  iter.NextValue();
  ASSERT_TRUE(iter.done());
}

TEST(SplitExpressionIterator, Tree) {
  /**     a
   *    / | \
   *   b  c  d
   *  / \
   * e   f
   **/
  Expression a;
  a.set_expression_type(Expression::ADD);
  Expression* b = a.add_arg();
  b->set_expression_type(Expression::ADD);
  Expression* c = a.add_arg();
  Expression* d = a.add_arg();
  Expression* e = b->add_arg();
  Expression* f = b->add_arg();

  a.mutable_size()->add_dim(1);
  b->mutable_size()->add_dim(2);
  c->mutable_size()->add_dim(3);
  d->mutable_size()->add_dim(4);
  e->mutable_size()->add_dim(5);
  f->mutable_size()->add_dim(6);

  SplitExpressionIterator iter(a);

  // e
  ASSERT_FALSE(iter.done());
  EXPECT_EQ(5, iter.leaf().size().dim(0));
  EXPECT_EQ(1, iter.chain().size().dim(0));
  EXPECT_EQ(2, iter.chain().arg(0).size().dim(0));
  EXPECT_EQ(5, iter.chain().arg(0).arg(0).size().dim(0));
  EXPECT_EQ(0, iter.chain().arg(0).arg(0).arg_size());

  // f
  iter.NextValue();
  ASSERT_FALSE(iter.done());
  EXPECT_EQ(6, iter.leaf().size().dim(0));
  EXPECT_EQ(1, iter.chain().size().dim(0));
  EXPECT_EQ(2, iter.chain().arg(0).size().dim(0));
  EXPECT_EQ(6, iter.chain().arg(0).arg(0).size().dim(0));
  EXPECT_EQ(0, iter.chain().arg(0).arg(0).arg_size());

  // c
  iter.NextValue();
  ASSERT_FALSE(iter.done());
  EXPECT_EQ(3, iter.leaf().size().dim(0));
  EXPECT_EQ(1, iter.chain().size().dim(0));
  EXPECT_EQ(3, iter.chain().arg(0).size().dim(0));
  EXPECT_EQ(0, iter.chain().arg(0).arg_size());

  // d
  iter.NextValue();
  ASSERT_FALSE(iter.done());
  EXPECT_EQ(4, iter.leaf().size().dim(0));
  EXPECT_EQ(1, iter.chain().size().dim(0));
  EXPECT_EQ(4, iter.chain().arg(0).size().dim(0));
  EXPECT_EQ(0, iter.chain().arg(0).arg_size());

  // Done
  iter.NextValue();
  ASSERT_TRUE(iter.done());
}

TEST(SplitExpressionIterator, Single_NoSplit) {
  /**     a
   *     / \
   *    b   c
   */

  Expression a;
  a.set_expression_type(Expression::MULTIPLY);
  Expression* b = a.add_arg();
  Expression* c = a.add_arg();

  a.mutable_size()->add_dim(1);
  b->mutable_size()->add_dim(2);
  c->mutable_size()->add_dim(3);

  SplitExpressionIterator iter(a);

  // a
  ASSERT_FALSE(iter.done());
  EXPECT_EQ(1, iter.leaf().size().dim(0));
  EXPECT_EQ(2, iter.leaf().arg(0).size().dim(0));
  EXPECT_EQ(3, iter.leaf().arg(1).size().dim(0));
  EXPECT_EQ(1, iter.chain().size().dim(0));
  EXPECT_EQ(2, iter.chain().arg(0).size().dim(0));
  EXPECT_EQ(3, iter.chain().arg(1).size().dim(0));

  iter.NextValue();
  ASSERT_TRUE(iter.done());
}

TEST(SplitExpressionIterator, Tree_NoSplit) {
  /**     a
   *    / | \
   *   b  c  d
   *  / \
   * e   f
   **/
  Expression a;
  a.set_expression_type(Expression::ADD);
  Expression* b = a.add_arg();
  b->set_expression_type(Expression::MULTIPLY);
  Expression* c = a.add_arg();
  Expression* d = a.add_arg();
  Expression* e = b->add_arg();
  Expression* f = b->add_arg();

  a.mutable_size()->add_dim(1);
  b->mutable_size()->add_dim(2);
  c->mutable_size()->add_dim(3);
  d->mutable_size()->add_dim(4);
  e->mutable_size()->add_dim(5);
  f->mutable_size()->add_dim(6);

  SplitExpressionIterator iter(a);

  // b
  ASSERT_FALSE(iter.done());
  EXPECT_EQ(2, iter.leaf().size().dim(0));
  EXPECT_EQ(5, iter.leaf().arg(0).size().dim(0));
  EXPECT_EQ(6, iter.leaf().arg(1).size().dim(0));
  EXPECT_EQ(1, iter.chain().size().dim(0));
  EXPECT_EQ(2, iter.chain().arg(0).size().dim(0));
  EXPECT_EQ(5, iter.chain().arg(0).arg(0).size().dim(0));
  EXPECT_EQ(6, iter.chain().arg(0).arg(1).size().dim(0));

  // c
  iter.NextValue();
  ASSERT_FALSE(iter.done());
  EXPECT_EQ(3, iter.leaf().size().dim(0));
  EXPECT_EQ(1, iter.chain().size().dim(0));
  EXPECT_EQ(3, iter.chain().arg(0).size().dim(0));
  EXPECT_EQ(0, iter.chain().arg(0).arg_size());

  // d
  iter.NextValue();
  ASSERT_FALSE(iter.done());
  EXPECT_EQ(4, iter.leaf().size().dim(0));
  EXPECT_EQ(1, iter.chain().size().dim(0));
  EXPECT_EQ(4, iter.chain().arg(0).size().dim(0));
  EXPECT_EQ(0, iter.chain().arg(0).arg_size());

  // Done
  iter.NextValue();
  ASSERT_TRUE(iter.done());
}
