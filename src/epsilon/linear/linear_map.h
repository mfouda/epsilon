#ifndef EPSILON_LINEAR_LINEAR_MAP_H
#define EPSILON_LINEAR_LINEAR_MAP_H

#include <memory>
#include <string>

#include <glog/logging.h>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

enum LinearMapImplType {
  DENSE_MATRIX,
  SPARSE_MATRIX,
  DIAGONAL_MATRIX,
  SCALAR_MATRIX,
  KRONECKER_PRODUCT,
  // only supports Apply()
  BASIC,
  NUM_LINEAR_MAP_IMPL_TYPES,
};

class LinearMapImpl {
 public:
  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DenseMatrix;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> DenseVector;
  typedef Eigen::SparseMatrix<Scalar> SparseMatrix;

  LinearMapImpl(LinearMapImplType type) : type_(type) {}
  virtual ~LinearMapImpl() {}

  virtual int m() const = 0;
  virtual int n() const = 0;
  virtual std::string DebugString() const = 0;
  virtual DenseMatrix AsDense() const = 0;
  virtual DenseVector Apply(const DenseVector& x) const = 0;

  virtual std::unique_ptr<LinearMapImpl> Transpose() const = 0;
  virtual std::unique_ptr<LinearMapImpl> Inverse() const = 0;

  LinearMapImplType type() const { return type_; }

 private:
  LinearMapImplType type_;
};

// A convenient wrapper around LinearMapImpl that can be passed by value,
// used with operator overloading, etc.
class LinearMap {
 public:
  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DenseMatrix;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> DenseVector;
  typedef Eigen::SparseMatrix<Scalar> SparseMatrix;

  LinearMap(std::unique_ptr<LinearMapImpl> impl)
      : impl_(std::move(impl)) {}

  static LinearMap Identity(int n);
  static LinearMap FromDense(DenseMatrix A);
  static LinearMap FromSparse(SparseMatrix A);

  // Accessor for implementation
  const LinearMapImpl& impl() const { return *impl_; }

  // These just wrap the values in impl
  LinearMap Inverse() const { return LinearMap(std::move(impl_->Inverse())); }
  LinearMap Transpose() const {
    return LinearMap(std::move(impl_->Transpose()));
  }

 private:
  std::unique_ptr<LinearMapImpl> impl_;
};

// Matrix-matrix multiply, add, subtract
// LinearMap operator-(LinearMap lhs, LinearMap rhs);
// LinearMap operator+(LinearMap lhs, LinearMap rhs);
LinearMap operator*(const LinearMap& lhs, const LinearMap& rhs);

// Matrix-vector multiply
template<typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> operator*(
    const LinearMap& lhs,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& rhs);

#endif  // EPSILON_LINEAR_LINEAR_MAP_H
