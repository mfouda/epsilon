#include "epsilon/vector/matrix_variant.h"

class DenseSolver : public MatrixVariant::Solver {
 public:
  DenseSolver(const MatrixVariant::DenseMatrix& A) {
    // TODO(mwytock): Handle failure more gracefully. This is especially
    // important for numerical issues where we may want to add epsI to diagonal
    // if LLT fails.
    llt_.compute(A);
    CHECK_EQ(Eigen::Success, llt_.info());
  }

  MatrixVariant::DenseVector solve(
      const MatrixVariant::DenseVector& b) override {
    return llt_.solve(b);
  }

 private:
  Eigen::LLT<MatrixVariant::DenseMatrix> llt_;
};

int MatrixVariant::rows() const {
  switch (type_) {
    case DENSE:
      return dense_.rows();
    case SPARSE:
      return sparse_.rows();
    case DIAGONAL:
      return diagonal_.rows();
    case SCALAR:
      return scalar_.n;
  }
  LOG(FATAL) << "unknown type: " << type_;
}

int MatrixVariant::cols() const {
  switch (type_) {
    case DENSE:
      return dense_.cols();
    case SPARSE:
      return sparse_.cols();
    case DIAGONAL:
      return diagonal_.cols();
    case SCALAR:
      return scalar_.n;
  }
  LOG(FATAL) << "unknown type: " << type_;
}

MatrixVariant MatrixVariant::transpose() const {
  switch (type_) {
    case DENSE:
      return MatrixVariant(dense_.transpose());
    case SPARSE:
      return MatrixVariant(static_cast<MatrixVariant::SparseMatrix>(
          sparse_.transpose()));
    case DIAGONAL:
    case SCALAR:
      return *this;
  }
  LOG(FATAL) << "unknown type: " << type_;
}

std::unique_ptr<MatrixVariant::Solver> MatrixVariant::inv() const {
  switch (type_) {
    case DENSE:
      return std::unique_ptr<Solver>(new DenseSolver(dense_));
    case SPARSE:
    case DIAGONAL:
    case SCALAR:
      LOG(FATAL) << "Not implemented";
  }
  LOG(FATAL) << "unknown type: " << type_;
}

MatrixVariant::DenseMatrix MatrixVariant::AsDense() const {
  switch (type_) {
    case MatrixVariant::DENSE:
      return dense_;
    case MatrixVariant::SPARSE:
      return static_cast<DenseMatrix>(sparse_);
    case MatrixVariant::DIAGONAL:
      return static_cast<DenseMatrix>(diagonal_);
    case MatrixVariant::SCALAR:
      return static_cast<DenseMatrix>(
            scalar_.alpha*DenseMatrix::Identity(scalar_.n, scalar_.n));
  }
  LOG(FATAL) << "unknown type: " << type_;
}

MatrixVariant& MatrixVariant::operator+=(const MatrixVariant& rhs) {
  CHECK_EQ(cols(), rhs.cols());
  CHECK_EQ(rows(), rhs.rows());
  const Type& t = type_;
  const Type& s = rhs.type_;

  if (t == DENSE) {
    if      (s == DENSE)    { dense_ += rhs.dense_; }
    else if (s == SPARSE)   { dense_ += rhs.sparse_; }
    else if (s == DIAGONAL) { dense_ += rhs.diagonal_; }
    else if (s == SCALAR)   {
      dense_ += rhs.scalar_.alpha*MatrixVariant::DenseMatrix::Identity(
          rhs.scalar_.n, rhs.scalar_.n);
    }
  }
  else if (t == SPARSE) {
    if (s == DENSE) {
      *this = MatrixVariant(
          (MatrixVariant::DenseMatrix(sparse_) + rhs.dense_).eval());
    }
    else if (s == SPARSE)   { sparse_ += rhs.sparse_; }
    else if (s == DIAGONAL) {
      sparse_ += DiagonalSparse(rhs.diagonal_.diagonal());
    }
    else if (s == SCALAR)   {
      sparse_ += rhs.scalar_.alpha*SparseIdentity(
          rhs.scalar_.n);
    }
  }

  return *this;
}

MatrixVariant& MatrixVariant::operator*=(const MatrixVariant& rhs) {
  CHECK_EQ(cols(), rhs.rows());
  const Type& t = type_;
  const Type& s = rhs.type_;

  if (t == DENSE) {
    if      (s == DENSE)    { dense_ *= rhs.dense_; }
    else if (s == SPARSE)   { dense_ *= rhs.sparse_; }
    else if (s == DIAGONAL) { dense_ *= rhs.diagonal_; }
    else if (s == SCALAR)   { dense_ *= rhs.scalar_.n; }
  }
  else if (t == SPARSE) {
    if (s == DENSE)         { *this =
          MatrixVariant((sparse_ * rhs.dense_).eval()); }
    else if (s == SPARSE)   { sparse_ = sparse_ * rhs.sparse_; }
    else if (s == DIAGONAL) {
      sparse_ = sparse_ * DiagonalSparse(rhs.diagonal_.diagonal());
    }
    else if (s == SCALAR)   { sparse_ *= rhs.scalar_.alpha; }
  }

  return *this;
}

MatrixVariant operator+(MatrixVariant lhs, const MatrixVariant& rhs) {
  lhs += rhs;
  return lhs;
}

MatrixVariant operator*(MatrixVariant lhs, const MatrixVariant& rhs) {
  lhs *= rhs;
  return lhs;
}

Eigen::VectorXd operator*(
    const MatrixVariant& lhs,
    const Eigen::VectorXd& rhs) {
  switch (lhs.type_) {
    case MatrixVariant::DENSE:
      return lhs.dense_*rhs;
    case MatrixVariant::SPARSE:
      return lhs.sparse_*rhs;
    case MatrixVariant::DIAGONAL:
      return lhs.diagonal_*rhs;
    case MatrixVariant::SCALAR:
      CHECK_EQ(lhs.scalar_.n, rhs.rows());
      return lhs.scalar_.alpha*rhs;
  }
  LOG(FATAL) << "unknown type: " << lhs.type_;
}
