#include <stdlib.h>

#include <memory>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "epsilon/affine/split.h"
#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression.h"
#include "epsilon/expression/expression_testutil.h"

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

  // c
  ASSERT_FALSE(iter.done());
  EXPECT_EQ(3, iter.leaf().size().dim(0));
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

  // f
  ASSERT_FALSE(iter.done());
  EXPECT_EQ(6, iter.leaf().size().dim(0));
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

TEST(SplitExpressionIterator, Tree_Negate) {
  /**    a
   *    / \
   *   b   f
   *   |
   *   c
   *  / \
   * d   e
   **/
  Expression a;
  a.set_expression_type(Expression::ADD);
  Expression* b = a.add_arg();
  b->set_expression_type(Expression::NEGATE);
  Expression* c = b->add_arg();
  c->set_expression_type(Expression::ADD);
  Expression* d = c->add_arg();
  Expression* e = c->add_arg();
  Expression* f = a.add_arg();

  a.mutable_variable()->set_variable_id("a");
  b->mutable_variable()->set_variable_id("b");
  c->mutable_variable()->set_variable_id("c");
  d->mutable_variable()->set_variable_id("d");
  e->mutable_variable()->set_variable_id("e");
  f->mutable_variable()->set_variable_id("f");

  SplitExpressionIterator iter(a);

  // d
  ASSERT_FALSE(iter.done());
  EXPECT_EQ("a", iter.chain().variable().variable_id());
  EXPECT_EQ("b", iter.chain().arg(0).variable().variable_id());
  EXPECT_EQ("c", iter.chain().arg(0).arg(0).variable().variable_id());
  EXPECT_EQ("d", iter.chain().arg(0).arg(0).arg(0).variable().variable_id());
  EXPECT_EQ("d", iter.leaf().variable().variable_id());

  // e
  iter.NextValue();
  ASSERT_FALSE(iter.done());
  EXPECT_EQ("a", iter.chain().variable().variable_id());
  EXPECT_EQ("b", iter.chain().arg(0).variable().variable_id());
  EXPECT_EQ("c", iter.chain().arg(0).arg(0).variable().variable_id());
  EXPECT_EQ("e", iter.chain().arg(0).arg(0).arg(0).variable().variable_id());
  EXPECT_EQ("e", iter.leaf().variable().variable_id());

  // f
  iter.NextValue();
  ASSERT_FALSE(iter.done());
  EXPECT_EQ("a", iter.chain().variable().variable_id());
  EXPECT_EQ("f", iter.chain().arg(0).variable().variable_id());
  EXPECT_EQ("f", iter.leaf().variable().variable_id());

  iter.NextValue();
  ASSERT_TRUE(iter.done());
}
