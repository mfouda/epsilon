
#include <Eigen/SparseCholesky>

#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"

#include "epsilon/util/string.h"
#include "epsilon/vector/vector_util.h"

namespace linear_map {

class SparseLDLImpl final : public LinearMapImpl {
public:
  SparseLDLImpl(SparseMatrix A)
      : LinearMapImpl(BASIC),
      solver_(A),
      transpose_(false) {
    CHECK_EQ(solver_.info(), Eigen::Success);
  }

  int m() const override { return solver_.matrixL().rows(); }
  int n() const override { return solver_.matrixL().rows(); }
  std::string DebugString() const override { return "SparseLDLImpl"; }

  DenseMatrix AsDense() const override {
    LOG(FATAL) << "Not implemented";
  }

  DenseVector Apply(const DenseVector& x) const override {
    if (transpose_)
      LOG(FATAL) << "Not implemented";
    return solver_.solve(x);
  }

  DenseMatrix ApplyMatrix(const DenseMatrix& x) const override {
    if (transpose_)
      LOG(FATAL) << "Not implemented";
    return solver_.solve(x);
  }

  LinearMapImpl* Transpose() const override {
    LOG(FATAL) << "Not implemented";
  }

  LinearMapImpl* Inverse() const override {
    LOG(FATAL) << "Not implemented";
  }

  bool operator==(const LinearMapImpl& other) const override {
    LOG(FATAL) << "Not implemented";
  }

private:
  Eigen::SimplicialLDLT<SparseMatrix> solver_;
  bool transpose_;
};

std::string SparseMatrixImpl::DebugString() const {
  return StringPrintf(
      "sparse matrix %d x %d\n%s",
      A_.rows(), A_.cols(),
      SparseMatrixDebugString(A_).c_str());
}

LinearMapImpl* SparseMatrixImpl::Inverse() const {
  // TODO(mwytock): Verify symmetry, fill-in
  CHECK_EQ(A_.rows(), A_.cols());

  VLOG(1) << "Factoring " << A_.rows() << " x " << A_.cols()
          << ", nnz=" << A_.nonZeros();

  double alpha;
  if (IsScalarMatrix(A_, &alpha)) {
    // Convert to scalar matrix
    std::unique_ptr<LinearMapImpl> impl(new ScalarMatrixImpl(n(), alpha));
    return impl->Inverse();
  } else {
    return new SparseLDLImpl(A_);
  }
}

bool SparseMatrixImpl::operator==(const LinearMapImpl& other) const {
  if (other.type() != SPARSE_MATRIX ||
      other.m() != m() ||
      other.n() != n())
    return false;

  // Sparse matrix equality not implemented in Eigen?
  // TODO(mwytock): Fix this
  return false;
}

}  // namespace linear_map
