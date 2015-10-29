#include "epsilon/linear/dense_matrix_impl.h"
#include "epsilon/linear/scalar_matrix_impl.h"
#include "epsilon/linear/sparse_matrix_impl.h"

#include "epsilon/util/string.h"
#include "epsilon/vector/vector_util.h"

namespace linear_map {

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
    // Convert to dense matrix
    std::unique_ptr<LinearMapImpl> impl(new DenseMatrixImpl(A_));
    return impl->Inverse();
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
