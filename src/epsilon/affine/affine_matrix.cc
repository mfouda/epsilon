#include "epsilon/affine/affine_matrix.h"
#include <unordered_map>

#include <glog/logging.h>

#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/vector/vector_file.h"
#include "epsilon/vector/vector_util.h"

// Convenice functions to handle zeros
#define ADD(A, B) (!(A).rows() ? (B) : (!(B).rows() ? (A) : (A)+(B)))
#define MULTIPLY(A, B) (!(A).rows() ? (A) : (!(B).rows() ? (B) : (A)*(B)))

namespace affine {

namespace {

Eigen::MatrixXd ReadConstant(const Expression& expr) {
  const int m = GetDimension(expr, 0);
  const int n = GetDimension(expr, 1);
  const Constant& c = expr.constant();
  if (c.data_location() == "")
    return Eigen::MatrixXd::Constant(m, n, c.scalar());

  VLOG(1) << "Read: " << c.data_location();
  std::unique_ptr<const Data> d = ReadSplitData(c.data_location());
  VLOG(1) << "Read done: " << c.data_location();
  return GetMatrixData(*d);
}

MatrixOperator Add(const Expression& expr) {
  CHECK(expr.arg_size() != 0);

  MatrixOperator op = BuildMatrixOperator(expr.arg(0));
  for (int i = 1; i < expr.arg_size(); i++) {
    MatrixOperator op_i = BuildMatrixOperator(expr.arg(i));
    if (!op.A.rows() && !op.B.rows()) {
      op.A = op_i.A;
      op.B = op_i.B;
    } else if (!op_i.A.rows() && !op_i.B.rows()) {
      // Do nothing
    } else if (IsMatrixEqual(op.A, op_i.A)) {
      op.B = ADD(op.B, op_i.B);
    } else if (IsMatrixEqual(op.B, op_i.B)) {
      op.A = ADD(op.A, op_i.A);
    } else {
      LOG(FATAL) << "Incompatible operators\n"
                 << "A1:\n" << MatrixDebugString(op_i.A)
                 << "B1:\n" << MatrixDebugString(op_i.B)
                 << "A2:\n" << MatrixDebugString(op.A)
                 << "B2:\n" << MatrixDebugString(op.B);
    }
    op.C = ADD(op.C, op_i.C);
  }
  return op;
}

MatrixOperator Multiply(const Expression& expr) {
  CHECK_EQ(2, expr.arg_size());
  CHECK_EQ(GetDimension(expr, 0), GetDimension(expr.arg(0), 0));
  CHECK_EQ(GetDimension(expr, 1), GetDimension(expr.arg(1), 1));
  CHECK_EQ(GetDimension(expr.arg(0), 1), GetDimension(expr.arg(1), 0));

  MatrixOperator lhs = BuildMatrixOperator(expr.arg(0));
  MatrixOperator rhs = BuildMatrixOperator(expr.arg(1));
  if (lhs.A.isZero()) {
    CHECK(lhs.B.isZero());
    rhs.A = MULTIPLY(lhs.C, rhs.A);
    rhs.C = MULTIPLY(lhs.C, rhs.C);
    return rhs;
  } else if (rhs.A.isZero()) {
    CHECK(rhs.B.isZero());
    lhs.B = MULTIPLY(lhs.B, rhs.C);
    lhs.C = MULTIPLY(lhs.C, rhs.C);
    return lhs;
  }

  LOG(FATAL) << "multiplying on both sides";
}

MatrixOperator Negate(const Expression& expr) {
  CHECK_EQ(1, expr.arg_size());
  MatrixOperator op = BuildMatrixOperator(expr.arg(0));
  if (!op.A.isZero() && !op.B.isZero()) {
    if (op.A.isIdentity())
      op.B *= -1;
    else
      op.A *= -1;
  }

  if (!op.C.isZero())
    op.C *= -1;

  return op;
}

MatrixOperator Index(const Expression& expr) {
  CHECK_EQ(1, expr.arg_size());
  MatrixOperator op = BuildMatrixOperator(expr.arg(0));

  CHECK_EQ(expr.key_size(), 2);
  CHECK_EQ(expr.key(0).step(), 1);
  CHECK_EQ(expr.key(1).step(), 1);
  const int m = expr.key(0).stop() - expr.key(0).start();
  const int n = expr.key(1).stop() - expr.key(1).start();

  if (op.A.rows() && op.B.rows()) {
    op.A = op.A.middleRows(expr.key(0).start(), m).eval();
    op.B = op.B.middleCols(expr.key(1).start(), n).eval();
  }
  if (op.C.rows()) {
    op.C = op.C.block(expr.key(0).start(), m, expr.key(1).start(), n).eval();
  }

  return op;
}

MatrixOperator Variable(const Expression& expr) {
  const int m = GetDimension(expr, 0);
  const int n = GetDimension(expr, 1);
  MatrixOperator op;
  op.A = Eigen::MatrixXd::Identity(m, m);
  op.B = Eigen::MatrixXd::Identity(n, n);
  return op;
}

MatrixOperator Constant(const Expression& expr) {
  MatrixOperator op;
  op.C = ReadConstant(expr);
  return op;
}

}  // namespace

typedef MatrixOperator(*AffineMatrixFunction)(
    const Expression& expr);
std::unordered_map<int, AffineMatrixFunction> kAffineMatrixFunctions = {
  {Expression::ADD, &Add},
  {Expression::CONSTANT, &Constant},
  {Expression::INDEX, &Index},
  {Expression::MULTIPLY, &Multiply},
  {Expression::NEGATE, &Negate},
  {Expression::VARIABLE, &Variable},
};

MatrixOperator BuildMatrixOperator(const Expression& expr) {
  VLOG(2) << "BuildMatrixOperator\n" << expr.DebugString();
  auto iter = kAffineMatrixFunctions.find(expr.expression_type());
  if (iter == kAffineMatrixFunctions.end()) {
    LOG(FATAL) << "No affine matrix function for "
               << Expression::Type_Name(expr.expression_type());
  }

  if (VLOG_IS_ON(2)) {
    MatrixOperator op = iter->second(expr);
    VLOG(2) << "BuildMatrixOperator "
            << Expression::Type_Name(expr.expression_type())
            << " returning\n"
            << "A:\n" << MatrixDebugString(op.A)
            << "B:\n" << MatrixDebugString(op.B)
            << "C:\n" << MatrixDebugString(op.C);
    return op;
  } else {
    return iter->second(expr);
  }
}

}  // namespace affine
