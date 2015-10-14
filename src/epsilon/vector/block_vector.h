
#ifndef EPSILON_VECTOR_BLOCK_VECTOR_H
#define EPSILON_VECTOR_BLOCK_VECTOR_H

#include <Eigen/Dense>

#include <map>

class BlockMatrix;

class BlockVector {
 public:
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> DenseVector;

  DenseVector& operator()(const std::string& key) { return data_[key]; };
  const DenseVector& operator()(const std::string& key) const;

  BlockVector& operator+=(const BlockVector& rhs);
  BlockVector& operator-=(const BlockVector& rhs);
  friend BlockVector operator*(const BlockMatrix& A, const BlockVector& x);

  double norm() const;

 private:
  void InsertOrAdd(const std::string& key, DenseVector value);

  std::map<std::string, DenseVector> data_;
};

BlockVector operator+(BlockVector lhs, const BlockVector& rhs);
BlockVector operator-(BlockVector lhs, const BlockVector& rhs);

#endif  // EPSILON_VECTOR_BLOCK_VECTOR_H
