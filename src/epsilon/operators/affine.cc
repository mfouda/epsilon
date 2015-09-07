#include "epsilon/operators/affine.h"

#include <memory>
#include <mutex>

#include "epsilon/util/logging.h"

#include <Eigen/SparseCore>

#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression.h"
#include "epsilon/file/file.h"
#include "epsilon/util/string.h"
#include "epsilon/util/vector.h"
#include "epsilon/util/vector_file.h"

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::LLT;

typedef Eigen::SparseMatrix<double> SparseXd;


DynamicMatrix ReadConstant(DynamicMatrix L, const Constant& c) {
  if (c.data_location() == "")
    return DynamicMatrix::FromDense(VectorXd::Constant(1, c.scalar()));

  CHECK(!c.transpose());  // TODO(mwytock): Remove this flag

  VLOG(1) << "Read: " << c.data_location();
  std::unique_ptr<const Data> d = ReadSplitData(c.data_location());
  VLOG(1) << "Read done: " << c.data_location();
  DynamicMatrix A = DynamicMatrix::FromDense(GetMatrixData(*d));
  A.ToVector();
  L.RightMultiply(A);
  return L;
}

void AppendBlockTriplets(
    const DynamicMatrix& input, int i, int j,
    std::vector<Eigen::Triplet<double>>* coeffs) {
  AppendBlockTriplets(input.AsDense(), i, j, coeffs);
}

DynamicMatrix ScalarMatrix(int n, double val) {
  std::vector<Eigen::Triplet<double> > coeffs;
  coeffs.reserve(n);
  for (int i = 0; i < n; i++) {
    coeffs.push_back(Eigen::Triplet<double>(i, i, val));
  }
  SparseXd A(n,n);
  A.setFromTriplets(coeffs.begin(), coeffs.end());
  return DynamicMatrix::FromSparse(A);
}

DynamicMatrix BlockDiag(DynamicMatrix A, int k) {
  if (k == 1)
    return A;

  return DynamicMatrix::FromSparse(BlockDiag(A.AsDense(), k));
}

namespace op {

DynamicMatrix Negate(const Expression& expr) {
  const int n = GetDimension(expr);
  SparseXd I(n, n);
  I.setIdentity();
  return DynamicMatrix::FromSparse(-1 * I);
}

DynamicMatrix Index(const Expression& expr) {
  const Expression& arg = GetOnlyArg(expr);
  CHECK_EQ(expr.size().dim_size(), 2);
  CHECK_EQ(2, expr.key_size());
  const int rows = GetDimension(arg, 0);
  const Slice& rkey = expr.key(0);
  const Slice& ckey = expr.key(1);

  std::vector<Eigen::Triplet<double> > coeffs;
  int k = 0;
  for (int j = ckey.start(); j < ckey.stop(); j += ckey.step()) {
    for (int i = rkey.start(); i < rkey.stop(); i += rkey.step()) {
      coeffs.push_back(Eigen::Triplet<double>(k++, j*rows + i, 1));
    }
  }

  SparseXd A(GetDimension(expr), GetDimension(arg));
  A.setFromTriplets(coeffs.begin(), coeffs.end());
  return DynamicMatrix::FromSparse(A);
}

DynamicMatrix Sum(const Expression& expr) {
  const int m = GetDimension(expr);
  const int n = GetDimension(GetOnlyArg(expr));
  return DynamicMatrix::FromDense(MatrixXd::Constant(m, n, 1));
}

DynamicMatrix Transpose(const Expression& expr) {
  CHECK_EQ(expr.size().dim_size(), 2);
  const int rows = expr.size().dim(0);
  const int cols = expr.size().dim(1);

  std::vector<Eigen::Triplet<double> > coeffs;
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i < rows; i++) {
      coeffs.push_back(Eigen::Triplet<double>(j*rows + i, i*cols + j, 1));
    }
  }
  SparseXd T(rows*cols, rows*cols);
  T.setFromTriplets(coeffs.begin(), coeffs.end());
  return DynamicMatrix::FromSparse(T);
}

DynamicMatrix Identity(const Expression& expr) {
  return DynamicMatrix::Identity(GetDimension(expr));
}

}  // namespace op

typedef DynamicMatrix(*LinearFunction)(
    const Expression& expr);
std::unordered_map<int, LinearFunction> kLinearFunctions = {
  {Expression::INDEX, &op::Index},
  {Expression::NEGATE, &op::Negate},
  {Expression::SUM, &op::Sum},
  {Expression::TRANSPOSE, &op::Transpose},
  {Expression::RESHAPE, &op::Identity},
};

DynamicMatrix LinearFunctionMatrix(const Expression& expr) {
  auto iter = kLinearFunctions.find(expr.expression_type());
  if (iter == kLinearFunctions.end()) {
    LOG(FATAL) << "No linear function for "
               << Expression::Type_Name(expr.expression_type());
  }
  return iter->second(expr);
}

// Top-down translation of expr into a set of linear operators applied to
// variables and a constaint term.
//
// Given an expression, generates the operators A, b in the expression:
//   Avec(X) + b
//
// L is pre-computed matrix for operators on the LHS of current expression.
void BuildAffineOperatorImpl(
    const Expression& expr,
    DynamicMatrix L,
    DynamicMatrix* A,
    DynamicMatrix* b) {
  VLOG(2) << "BuildAffineOperatorImpl L=(" << L.rows() << "," << L.cols() << ")\n"
          << expr.DebugString();

  if (expr.expression_type() == Expression::CONSTANT) {
    b->Add(ReadConstant(L, expr.constant()));
  } else if (expr.expression_type() == Expression::VARIABLE) {
    A->Add(L, 0, expr.variable().offset());
  } else if (expr.expression_type() == Expression::ADD) {
    for (const Expression& arg : expr.arg())
      BuildAffineOperatorImpl(arg, L, A, b);
  } else if (expr.expression_type() == Expression::MULTIPLY) {
    // Assume that the first n-1 are constants
    for (int i = 0; i < expr.arg_size() - 1; i++) {
      DynamicMatrix Ai;
      DynamicMatrix bi = DynamicMatrix::Zero(GetDimension(expr.arg(i)), 1);
      CHECK(Ai.is_zero());
      BuildAffineOperatorImpl(
          expr.arg(i),
          DynamicMatrix::FromSparse(SparseIdentity(GetDimension(expr.arg(i)))),
          &Ai, &bi);
      bi.ToMatrix(GetDimension(expr.arg(i), 0), GetDimension(expr.arg(i), 1));
      L.RightMultiply(BlockDiag(bi, GetDimension(expr, 1)));
    }
    BuildAffineOperatorImpl(expr.arg(expr.arg_size() - 1), L, A, b);
  } else if (expr.expression_type() == Expression::MULTIPLY_ELEMENTWISE) {
    // Assume that the first n-1 are constants
    for (int i = 0; i < expr.arg_size() - 1; i++) {
      DynamicMatrix Ai;
      DynamicMatrix bi = DynamicMatrix::Zero(GetDimension(expr.arg(i)), 1);
      CHECK(Ai.is_zero());
      BuildAffineOperatorImpl(
          expr.arg(i),
          DynamicMatrix::FromSparse(SparseIdentity(GetDimension(expr.arg(i)))),
          &Ai, &bi);
      L.RightMultiply(
          DynamicMatrix::FromSparse(DiagonalSparse(bi.AsDense())));
    }
    BuildAffineOperatorImpl(expr.arg(expr.arg_size() - 1), L, A, b);
  } else if (expr.expression_type() == Expression::HSTACK) {
    int offset = 0;
    for (const Expression& arg : expr.arg()) {
      int mi = GetDimension(arg);
      BuildAffineOperatorImpl(
          arg, L.GetColumns(offset, mi), A, b);
      offset += mi;
    }
  } else if (expr.expression_type() == Expression::VSTACK) {
    const int m = GetDimension(expr, 0);
    int offset = 0;
    for (int i = 0; i < expr.arg_size(); i++) {
      const Expression& arg = expr.arg(i);
      const int mi = GetDimension(arg, 0);
      const int ni = GetDimension(arg, 1);

      DynamicMatrix Li(L.rows(), GetDimension(arg));
      for (int j = 0; j < ni; j++) {
        Li.Add(L.GetColumns(offset + j*m, mi), 0, j*mi);
      }
      BuildAffineOperatorImpl(arg, Li, A, b);
      offset += mi;
    }
  } else {
    // Unary linear operator
    CHECK_EQ(1, expr.arg_size());
    L.RightMultiply(LinearFunctionMatrix(expr));
    BuildAffineOperatorImpl(expr.arg(0), L, A, b);
  }

  VLOG(2) << "BuildAffineOperatorImpl "
          << Expression::Type_Name(expr.expression_type())
          << " done";
}

class LinearExpressionOperatorImpl : public OperatorImpl {
 public:
  LinearExpressionOperatorImpl(
      const Expression& expr, bool transpose)
      : transpose_(transpose) {
    DynamicMatrix b;

    const int m = GetDimension(expr);

    BuildAffineOperatorImpl(
        expr, DynamicMatrix::Identity(m), &A_, &b);
    CHECK(b.is_zero());
  }

  virtual void Apply(const VectorXd& x, VectorXd* y) {
    if (transpose_)
      A_.ApplyTranspose(x, y);
    else
      A_.Apply(x, y);
  }

  virtual void ToMatrix(MatrixXd* A) {
    *A = A_.AsDense();
  }

 private:
  bool transpose_;
  DynamicMatrix A_;
};

class LinearProjectionOperatorImpl : public OperatorImpl {
 public:
  LinearProjectionOperatorImpl(const MatrixXd& A)
      : A_(A), llt_(GetFormToFactor(A)) {}

  virtual void Apply(const VectorXd& x, VectorXd* y) {
    // Input is interpreted as x = [c; d] with c in R^n and d in R^m
    y->resize(x.size());
    const int m = A_.rows();
    const int n = A_.cols();
    if (m > n) {
      y->head(n) = llt_.solve(x.head(n) + A_.transpose()*x.tail(m));
      y->tail(m) = A_*y->head(n);
    } else {
      y->tail(m) = llt_.solve(A_*x.head(n) + A_*A_.transpose()*x.tail(m));
      y->head(n) = x.head(n) + A_.transpose()*(x.tail(m) - y->tail(m));
    }
  }

  virtual void ToMatrix(MatrixXd* A) {
    LOG(FATAL) << "Not implemented";
  }

 private:
  MatrixXd GetFormToFactor(const MatrixXd& A) {
    const int m = A.rows();
    const int n = A.cols();
    if (m > n) {
      return MatrixXd::Identity(n,n) + A.transpose()*A;
    } else {
      return MatrixXd::Identity(m,m) + A*A.transpose();
    }
  }

  MatrixXd A_;
  LLT<MatrixXd> llt_;
};

OperatorImpl* BuildLinearExpressionOperator(
    const Expression& input, bool transpose) {
  VLOG(3) << "BuildOperator " << input.DebugString();
  return new LinearExpressionOperatorImpl(input, transpose);
}

OperatorImpl* BuildLinearProjectionOperator(const MatrixXd& A) {
  return new LinearProjectionOperatorImpl(A);
}

OperatorImpl* BuildLinearProjectionOperator(const SparseXd& A) {
  return new LinearProjectionOperatorImpl(static_cast<MatrixXd>(A));
}

void BuildSparseAffineOperator(
    const Expression& expr,
    int n,
    int i,
    std::vector<Eigen::Triplet<double>>* A_coeffs,
    VectorXd* b) {
  const int m = GetDimension(expr);
  DynamicMatrix A_ = DynamicMatrix::Zero(m, n);
  DynamicMatrix b_ = DynamicMatrix::Zero(m, 1);

  BuildAffineOperatorImpl(
      expr, DynamicMatrix::FromSparse(SparseIdentity(m)), &A_, &b_);
  AppendBlockTriplets(A_, i, 0, A_coeffs);
  if (b_.is_zero()) {
    b->segment(i, m) = VectorXd::Zero(m);
  } else {
    b->segment(i, m) = b_.AsDense();
  }
}

void BuildAffineOperator(
    const Expression& expr,
    int n,
    int i,
    MatrixXd* A,
    VectorXd* b) {
  const int m = GetDimension(expr);
  DynamicMatrix A_ = DynamicMatrix::Zero(m, n);
  DynamicMatrix b_ = DynamicMatrix::Zero(m, 1);

  BuildAffineOperatorImpl(
      expr, DynamicMatrix::FromSparse(SparseIdentity(m)), &A_, &b_);
  if (A_.is_zero()) {
    A->middleRows(i, m) = MatrixXd::Zero(m, n);
  } else {
    A->middleRows(i, m) = A_.AsDense();
  }

  if (b_.is_zero()) {
    b->segment(i, m) = VectorXd::Zero(m);
  } else {
    b->segment(i, m) = b_.AsDense();
  }
}

void BuildAffineOperator(
    const Expression& expr,
    DynamicMatrix* A,
    DynamicMatrix* b) {
  const int m = GetDimension(expr);
  BuildAffineOperatorImpl(
      expr, DynamicMatrix::FromSparse(SparseIdentity(m)), A, b);
}
