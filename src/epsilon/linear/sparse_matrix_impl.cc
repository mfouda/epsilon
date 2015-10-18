#include "epsilon/linear/sparse_matrix_impl.h"

#include "epsilon/util/string.h"

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
  Eigen::SimplicialLDLT<LinearMap::SparseMatrix> ldlt_;
  ldlt_.compute(A_);
  CHECK_EQ(Eigen::Success, ldlt_.info());

  // TODO(mwytock): This should probably do something different like form a
  // LinearMap implementation based on backsolves. Maybe a subclass of
  // SparseMatrixImpl?
  VLOG(1) << "Forming inverse explicitly";
  LinearMap::SparseMatrix A_inv = ldlt_.solve(SparseIdentity(A_.rows()));
  VLOG(1) << "Inverse, nnz=" << A_inv.nonZeros();
  return new SparseMatrixImpl(A_inv);
}
