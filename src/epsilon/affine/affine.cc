#include "epsilon/affine/affine.h"

#include <memory>
#include <mutex>

#include <glog/logging.h>

#include <Eigen/SparseCore>

#include "epsilon/affine/split.h"
#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/file/file.h"
#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/diagonal_matrix_impl.h"
#include "epsilon/linear/kronecker_product_impl.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"
#include "epsilon/util/string.h"
#include "epsilon/vector/vector_file.h"
#include "epsilon/vector/vector_util.h"

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::LLT;

typedef Eigen::SparseMatrix<double> SparseXd;


DynamicMatrix ReadConstant(DynamicMatrix L, const Constant& c) {
  DynamicMatrix A;
  if (c.data_location() == "") {
    A = DynamicMatrix::FromDense(VectorXd::Constant(L.cols(), c.scalar()));
  } else {
    VLOG(2) << "Read: " << c.data_location();
    std::unique_ptr<const Data> d = ReadSplitData(c.data_location());
    VLOG(2) << "Read done: " << c.data_location();
    A = DynamicMatrix::FromDense(GetMatrixData(*d));
    A.ToVector();
  }

  L.RightMultiply(A);
  return L;
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
  int n = 0;
  for (const Expression& arg : expr.arg())
    n += GetDimension(arg);
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
    const VariableOffsetMap& offset_map,
    DynamicMatrix L,
    DynamicMatrix* A,
    DynamicMatrix* b) {
  VLOG(2) << "BuildAffineOperatorImpl L=(" << L.rows() << "," << L.cols() << ")\n"
          << expr.DebugString();

  if (expr.expression_type() == Expression::CONSTANT) {
    if (b != nullptr)
      b->Add(ReadConstant(L, expr.constant()));
  } else if (expr.expression_type() == Expression::VARIABLE) {
    const std::string var_id = expr.variable().variable_id();
    if (A != nullptr && offset_map.Contains(var_id))
      A->Add(L, 0, offset_map.Get(var_id));
  } else if (expr.expression_type() == Expression::ADD) {
    for (const Expression& arg : expr.arg())
      BuildAffineOperatorImpl(arg, offset_map, L, A, b);
  } else if (expr.expression_type() == Expression::MULTIPLY) {
    // Assume that the first n-1 are constants
    for (int i = 0; i < expr.arg_size() - 1; i++) {
      DynamicMatrix Ai;
      DynamicMatrix bi = DynamicMatrix::Zero(GetDimension(expr.arg(i)), 1);
      CHECK(Ai.is_zero());
      BuildAffineOperatorImpl(
          expr.arg(i),
          offset_map,
          DynamicMatrix::FromSparse(SparseIdentity(GetDimension(expr.arg(i)))),
          &Ai, &bi);
      bi.ToMatrix(GetDimension(expr.arg(i), 0), GetDimension(expr.arg(i), 1));
      L.RightMultiply(BlockDiag(bi, GetDimension(expr, 1)));
    }
    BuildAffineOperatorImpl(expr.arg(expr.arg_size() - 1), offset_map, L, A, b);
  } else if (expr.expression_type() == Expression::MULTIPLY_ELEMENTWISE) {
    // Assume that the first n-1 are constants
    for (int i = 0; i < expr.arg_size() - 1; i++) {
      DynamicMatrix Ai;
      DynamicMatrix bi = DynamicMatrix::Zero(GetDimension(expr.arg(i)), 1);
      CHECK(Ai.is_zero());
      BuildAffineOperatorImpl(
          expr.arg(i),
          offset_map,
          DynamicMatrix::FromSparse(SparseIdentity(GetDimension(expr.arg(i)))),
          &Ai, &bi);
      L.RightMultiply(
          DynamicMatrix::FromSparse(DiagonalSparse(bi.AsDense())));
    }
    BuildAffineOperatorImpl(expr.arg(expr.arg_size() - 1), offset_map, L, A, b);
  } else if (expr.expression_type() == Expression::HSTACK) {
    int offset = expr.stack_params().offset() * GetDimension(expr, 0);
    for (const Expression& arg : expr.arg()) {
      int mi = GetDimension(arg);
      BuildAffineOperatorImpl(
          arg, offset_map, L.GetColumns(offset, mi), A, b);
      offset += mi;
    }
  } else if (expr.expression_type() == Expression::VSTACK) {
    const int m = GetDimension(expr, 0);
    int offset = expr.stack_params().offset();
    for (int i = 0; i < expr.arg_size(); i++) {
      const Expression& arg = expr.arg(i);
      const int mi = GetDimension(arg, 0);
      const int ni = GetDimension(arg, 1);

      DynamicMatrix Li(L.rows(), GetDimension(arg));
      for (int j = 0; j < ni; j++) {
        Li.Add(L.GetColumns(offset + j*m, mi), 0, j*mi);
      }
      BuildAffineOperatorImpl(arg, offset_map, Li, A, b);
      offset += mi;
    }
  } else {
    // Unary linear operator
    CHECK_EQ(1, expr.arg_size())
        << "More than one argument: " << expr.DebugString();
    L.RightMultiply(LinearFunctionMatrix(expr));
    BuildAffineOperatorImpl(expr.arg(0), offset_map, L, A, b);
  }

  VLOG(2) << "BuildAffineOperatorImpl "
          << Expression::Type_Name(expr.expression_type())
          << " done";
}

void BuildAffineOperator(
    const Expression& expr,
    const VariableOffsetMap& offsets,
    DynamicMatrix* A,
    DynamicMatrix* b) {
  const int m = GetDimension(expr);
  BuildAffineOperatorImpl(
      expr, offsets, DynamicMatrix::FromSparse(SparseIdentity(m)), A, b);
}

void GetDiagonalAffineOperator(
    const Expression& expr,
    const VariableOffsetMap& var_map,
    Eigen::VectorXd* a,
    Eigen::VectorXd* b_out) {
  const int m = GetDimension(expr);
  const int n = var_map.n();
  CHECK_EQ(m, n);

  DynamicMatrix A = DynamicMatrix::Zero(m, n);
  DynamicMatrix b = DynamicMatrix::FromDense(Eigen::VectorXd::Zero(m));
  BuildAffineOperator(expr, var_map, &A, &b);

  CHECK(A.is_sparse() && IsDiagonal(A.sparse()));
  *a = A.sparse().diagonal();
  *b_out = b.AsDense();
}

void GetScalarAffineOperator(
    const Expression& expr,
    const VariableOffsetMap& var_map,
    double* alpha,
    Eigen::VectorXd* b_out) {

  const int m = GetDimension(expr);
  const int n = var_map.n();
  CHECK_EQ(m, n) << "\n" << expr.DebugString();

  DynamicMatrix A = DynamicMatrix::Zero(m, n);
  DynamicMatrix b = DynamicMatrix::FromDense(Eigen::VectorXd::Zero(m));
  BuildAffineOperator(expr, var_map, &A, &b);

  CHECK(A.is_sparse());
  *alpha = A.sparse().coeff(0, 0);
  *b_out = b.AsDense();
}

SparseXd GetSparseAffineOperator(
    const Expression& expr,
    const VariableOffsetMap& var_map) {
  const int m = GetDimension(expr);
  const int n = var_map.n();
  DynamicMatrix A = DynamicMatrix::Zero(m, n);
  DynamicMatrix b = DynamicMatrix::FromDense(Eigen::VectorXd::Zero(m));
  BuildAffineOperator(expr, var_map, &A, &b);

  CHECK(b.is_zero());
  CHECK(A.is_sparse());
  return A.sparse();
}

SparseXd GetProjection(
    const VariableOffsetMap& a, const VariableOffsetMap& b) {
  std::vector<Eigen::Triplet<double> > coeffs;
  for (auto iter : b.offsets()) {
    const int i = iter.second;
    const int j = a.Get(iter.first);
    for (int k = 0; k < a.Size(iter.first); k++) {
      coeffs.push_back(Eigen::Triplet<double>(i+k, j+k, 1));
    }
  }
  SparseXd P(b.n(), a.n());
  P.setFromTriplets(coeffs.begin(), coeffs.end());
  return P;
}

namespace affine {

void BuildAffineOperatorImpl(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b);

// Utility function used by Multiply()
// TODO(mwytock): Should support non-dense constants.
Eigen::MatrixXd EvalConstant(
    const Expression& expr) {
  BlockVector b;
  BuildAffineOperatorImpl(
      expr,
      "_",
      linear_map::LinearMap::Identity(GetDimension(expr)),
      nullptr,
      &b);
  return ToMatrix(b("_"), GetDimension(expr,0), GetDimension(expr,1));
}

void Negate(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b) {
  BuildAffineOperatorImpl(
      GetOnlyArg(expr),
      row_key,
      // TODO(mwytock): We use scalar multiplication here instead of explicitly
      // mulitplying by a linear map to handle promotion. We should instead
      // handle this as a preprocessing step.
      -1*L,
      A, b);
}

void Sum(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b) {
  BuildAffineOperatorImpl(
      GetOnlyArg(expr),
      row_key,
      L*linear_map::LinearMap(new linear_map::DenseMatrixImpl(
          linear_map::DenseMatrixImpl::DenseMatrix::Constant(
              1, GetDimension(GetOnlyArg(expr)), 1))),
      A, b);
}

void Multiply(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b) {
  CHECK_EQ(2, expr.arg_size());

  if (expr.arg(0).curvature().curvature_type() == Curvature::CONSTANT &&
      GetDimension(expr, 1) == 1) {
    // Ax, get constant, handling promotion if necessary
    Eigen::MatrixXd C0 = EvalConstant(expr.arg(0));
    linear_map::LinearMap C;
    if (C0.rows() == 1 && GetDimension(expr, 0) != 1) {
      C = linear_map::LinearMap(new linear_map::ScalarMatrixImpl(GetDimension(expr, 0), C0(0,0)));
    } else {
      C = linear_map::LinearMap(new linear_map::DenseMatrixImpl(C0));
    }
    BuildAffineOperatorImpl(expr.arg(1), row_key, L*C, A, b);
  } else if (expr.arg(0).curvature().curvature_type() == Curvature::CONSTANT) {
    // AX
    BuildAffineOperatorImpl(
        expr.arg(1),
        row_key,
        L*linear_map::LinearMap(new linear_map::KroneckerProductImpl(
            new linear_map::ScalarMatrixImpl(GetDimension(expr, 1), 1),
            new linear_map::DenseMatrixImpl(EvalConstant(expr.arg(0))))),
        A, b);
  } else if (expr.arg(1).curvature().curvature_type() == Curvature::CONSTANT) {
    // XA
    BuildAffineOperatorImpl(
        expr.arg(0),
        row_key,
        L*linear_map::LinearMap(new linear_map::KroneckerProductImpl(
            new linear_map::DenseMatrixImpl(EvalConstant(expr.arg(1)).transpose()),
            new linear_map::ScalarMatrixImpl(GetDimension(expr, 0), 1))),
        A, b);
  } else {
    LOG(FATAL) << "Multiplying two non constants";
  }
}

void MultiplyElementwise(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b) {
  CHECK_EQ(2, expr.arg_size());

  const Expression* C_expr;
  const Expression* X_expr;
  if (expr.arg(0).curvature().curvature_type() == Curvature::CONSTANT) {
    C_expr = &expr.arg(0);
    X_expr = &expr.arg(1);
  } else if (expr.arg(0).curvature().curvature_type() == Curvature::CONSTANT) {
    C_expr = &expr.arg(1);
    X_expr = &expr.arg(0);
  } else {
    LOG(FATAL) << "Multiplying two non constants";
  }

  linear_map::DiagonalMatrixImpl::DiagonalMatrix C(GetDimension(*C_expr));
  C.diagonal() = ToVector(EvalConstant(*C_expr));
  BuildAffineOperatorImpl(
      *X_expr,
      row_key,
      L*linear_map::LinearMap(new linear_map::DiagonalMatrixImpl(C)),
      A, b);
}

void Index(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b) {
  const Expression& arg = GetOnlyArg(expr);
  CHECK_EQ(expr.size().dim_size(), 2);
  CHECK_EQ(2, expr.key_size());
  const int rows = GetDimension(arg, 0);
  const Slice& rkey = expr.key(0);
  const Slice& ckey = expr.key(1);

  SparseXd P(GetDimension(expr), GetDimension(arg));
  {
    std::vector<Eigen::Triplet<double> > coeffs;
    int k = 0;
    for (int j = ckey.start(); j < ckey.stop(); j += ckey.step()) {
      for (int i = rkey.start(); i < rkey.stop(); i += rkey.step()) {
        coeffs.push_back(Eigen::Triplet<double>(k++, j*rows + i, 1));
      }
    }
    P.setFromTriplets(coeffs.begin(), coeffs.end());
  }

  BuildAffineOperatorImpl(
      GetOnlyArg(expr),
      row_key,
      L*linear_map::LinearMap(new linear_map::SparseMatrixImpl(P)),
      A, b);
}

void Add(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b) {
  for (const Expression& arg : expr.arg())
    BuildAffineOperatorImpl(arg, row_key, L, A, b);
}

void Variable(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b) {
  A->InsertOrAdd(row_key, expr.variable().variable_id(), L);
}

void Constant(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b) {
  Eigen::VectorXd b_dense;
  const ::Constant& c = expr.constant();
  if (c.data_location() == "") {
    // Handle promotion if necessary by using L
    b_dense = Eigen::VectorXd::Constant(L.impl().n(), c.scalar());
  } else {
    VLOG(1) << "Read: " << c.data_location();
    std::unique_ptr<const Data> d = ReadSplitData(c.data_location());
    VLOG(1) << "Read done: " << c.data_location();
    b_dense = ToVector(GetMatrixData(*d));
  }

  b->InsertOrAdd(row_key, L*b_dense);
}

void Transpose(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b) {
  const int rows = GetDimension(expr, 0);
  const int cols = GetDimension(expr, 1);

  SparseXd T(rows*cols, rows*cols);
  {
    std::vector<Eigen::Triplet<double> > coeffs;
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        coeffs.push_back(Eigen::Triplet<double>(j*rows + i, i*cols + j, 1));
      }
    }
    T.setFromTriplets(coeffs.begin(), coeffs.end());
  }

  BuildAffineOperatorImpl(
      GetOnlyArg(expr),
      row_key,
      L*linear_map::LinearMap(new linear_map::SparseMatrixImpl(T)),
      A, b);
}

void LinearMap(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b) {
  BuildAffineOperatorImpl(
      GetOnlyArg(expr),
      row_key,
      L*linear_map::BuildLinearMap(expr.linear_map()),
      A, b);
}

typedef void(*LinearFunction)(
    const Expression&,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b);

std::unordered_map<int, LinearFunction> kLinearFunctions = {
  //{Expression::HSTACK, &HStack},
  //{Expression::VSTACK, &VStack},
  {Expression::ADD, &Add},
  {Expression::CONSTANT, &Constant},
  {Expression::MULTIPLY, &Multiply},
  {Expression::VARIABLE, &Variable},
  {Expression::LINEAR_MAP, &LinearMap},
};

void BuildAffineOperatorImpl(
    const Expression& expr,
    const std::string& row_key,
    linear_map::LinearMap L,
    BlockMatrix* A,
    BlockVector* b) {
  VLOG(2) << "BuildAffineOperatorImpl\n"
          << "L: " << L.impl().DebugString() << "\n"
          << expr.DebugString();

  auto iter = kLinearFunctions.find(expr.expression_type());
  if (iter == kLinearFunctions.end()) {
    LOG(FATAL) << "No linear function for "
               << Expression::Type_Name(expr.expression_type());
  }
  iter->second(expr, row_key, L, A, b);
}

void BuildAffineOperator(
    const Expression& expr,
    const std::string& row_key,
    BlockMatrix* A,
    BlockVector* b) {
  BuildAffineOperatorImpl(
      expr, row_key, linear_map::LinearMap::Identity(GetDimension(expr)), A, b);
}

}  // affine
